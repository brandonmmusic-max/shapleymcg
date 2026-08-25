from __future__ import annotations

import hashlib
import json
import os
import copy
from pathlib import Path

import numpy as np
import pytest

from quant_pipeline.campaign.runner import (
    CampaignRunner,
    EventJournal,
    StageRequest,
    StageResult,
    audit_campaign,
    create_plan,
    status_campaign,
)
from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json


REVISION = "a" * 40


def _artifact_path(request: StageRequest, stage_id: str) -> Path:
    return request.campaign_dir / request.dependency_artifacts[stage_id]["path"]


def _softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - np.max(value, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _kld(reference: np.ndarray, candidate: np.ndarray) -> float:
    p = _softmax(reference.astype(np.float64))
    q = _softmax(candidate.astype(np.float64))
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=-1)))


def _quantize(value: np.ndarray, bits: int) -> np.ndarray:
    limit = (1 << (bits - 1)) - 1
    scale = float(np.max(np.abs(value))) / limit
    return np.clip(np.rint(value / scale), -limit, limit).astype(np.int8).astype(np.float64) * scale


class TinyMoeAdapter:
    """Deterministic numeric adapter used only to exercise the real runner contract."""

    def __init__(
        self,
        *,
        fail_once: str | None = None,
        gate_passes: bool = True,
        confirmation_passes: bool = True,
    ):
        self.calls: dict[str, int] = {}
        self.fail_once = fail_once
        self.gate_passes = gate_passes
        self.confirmation_passes = confirmation_passes

    def identity(self):
        # Runtime counters are deliberately absent: they do not alter scientific behavior.
        return {
            "name": "deterministic-tiny-moe",
            "version": 1,
            "gate_passes": self.gate_passes,
            "confirmation_passes": self.confirmation_passes,
        }

    def preflight(self, plan):
        return {
            "ok": True,
            "local_only": True,
            "remote_endpoints": [],
            "gpu": {"required": False, "count": 0},
            "storage": {
                "retention_mode": plan["definition"]["retention_mode"],
                "estimated_peak_bytes": 1,
                "available_bytes": 1_000_000,
                "safety_margin_bytes": 1,
            },
            "software": {"numpy": np.__version__},
        }

    def _source(self, request: StageRequest) -> tuple[np.ndarray, np.ndarray]:
        source = Path(request.static_inputs["source_checkpoint"]["path"])
        data = np.load(source)
        return data["weights"].astype(np.float64), data["features"].astype(np.float64)

    def _encoded_weights(self, request: StageRequest) -> np.ndarray:
        weights, _ = self._source(request)
        encoded = [
            (stage_id, artifact)
            for stage_id, artifact in request.dependency_artifacts.items()
            if stage_id.startswith("causal_encode.")
        ]
        if not encoded:
            return weights
        _, latest = max(encoded, key=lambda item: int(item[0].rsplit("_", 1)[1]))
        return np.load(request.campaign_dir / latest["path"] / "installed-checkpoint.npy")

    def run(self, request: StageRequest) -> StageResult:
        self.calls[request.stage_id] = self.calls.get(request.stage_id, 0) + 1
        if self.fail_once == request.stage_id and self.calls[request.stage_id] == 1:
            raise RuntimeError("injected interruption")
        weights, features = self._source(request)
        kind = request.kind
        metadata: dict = {"kind": kind}
        if kind == "identity":
            write_json(request.output_dir / "identity.json", {"shape": list(weights.shape), "source": request.static_inputs["source_checkpoint"]["sha256"]})
        elif kind == "teacher_capture":
            hidden = features.copy()
            for matrix in weights:
                hidden = np.tanh(hidden @ matrix)
            np.save(request.output_dir / "teacher-logits.npy", hidden)
            metadata["mean"] = float(hidden.mean())
            metadata["teacher_reference_sha256"] = sha256_file(request.output_dir / "teacher-logits.npy")
        elif kind == "fit_capture":
            np.save(request.output_dir / "routed-inputs.npy", features)
            metadata["rows"] = int(features.shape[0])
        elif kind == "fit":
            routed = np.load(_artifact_path(request, "fit_capture") / "routed-inputs.npy")
            covariance = routed.T @ routed / routed.shape[0]
            np.save(request.output_dir / "covariance.npy", covariance)
            metadata["trace"] = float(np.trace(covariance))
            metadata["transient_files"] = ["covariance.npy"]
        elif kind == "candidates":
            covariance = np.load(_artifact_path(request, "fit") / "covariance.npy")
            rows = []
            for layer, matrix in enumerate(weights):
                for bits in (2, 3, 4):
                    reconstructed = _quantize(matrix, bits)
                    residual = matrix - reconstructed
                    rows.append(
                        {
                            "layer": layer,
                            "bits": bits,
                            "stored_bytes": int(np.ceil(matrix.size * bits / 8)),
                            "damage": float(np.trace(residual.T @ covariance @ residual)),
                            "reconstruction_sha256": hashlib.sha256(reconstructed.tobytes()).hexdigest(),
                        }
                    )
            write_json(request.output_dir / "candidates.json", {"rows": rows})
            metadata["candidate_count"] = len(rows)
        elif kind == "attribution":
            candidates = json.loads((_artifact_path(request, "candidates") / "candidates.json").read_text())["rows"]
            by_layer = {layer: min(row["damage"] for row in candidates if row["layer"] == layer) for layer in range(len(weights))}
            total = sum(by_layer.values())
            write_json(request.output_dir / "attribution.json", {"layers": by_layer, "total": total})
            metadata["closed_damage"] = float(total)
        elif kind == "allocation":
            candidates = json.loads((_artifact_path(request, "candidates") / "candidates.json").read_text())["rows"]
            choices = [min((row for row in candidates if row["layer"] == layer), key=lambda row: (row["damage"], row["stored_bytes"])) for layer in range(len(weights))]
            write_json(request.output_dir / "allocation.json", {"choices": choices})
            metadata["stored_bytes"] = sum(row["stored_bytes"] for row in choices)
        elif kind == "confirmation":
            allocation = json.loads(
                (_artifact_path(request, "allocation") / "allocation.json").read_text()
            )
            reconstructed = weights.copy()
            for choice in allocation["choices"]:
                reconstructed[int(choice["layer"])] = _quantize(
                    weights[int(choice["layer"])], int(choice["bits"])
                )
            reference = features.copy()
            student = features.copy()
            for source_matrix, candidate_matrix in zip(weights, reconstructed, strict=True):
                reference = np.tanh(reference @ source_matrix)
                student = np.tanh(student @ candidate_matrix)
            value = _kld(reference, student)
            reference_path = request.output_dir / "confirmation-reference.npy"
            capture_path = request.output_dir / "confirmation-student.npy"
            np.save(reference_path, reference)
            np.save(capture_path, student)
            threshold = (
                value + 1e-9
                if self.confirmation_passes
                else max(0.0, value - 1e-9)
            )
            metadata["gate"] = {
                "passed": self.confirmation_passes,
                "metric": "kld",
                "value": value,
                "threshold": threshold,
                "reference_sha256": sha256_file(reference_path),
                "capture_sha256": sha256_file(capture_path),
            }
            metadata["gate_files"] = {
                "reference": reference_path.name,
                "capture": capture_path.name,
            }
        elif kind == "causal_fit_capture":
            installed = self._encoded_weights(request)
            hidden = features.copy()
            for matrix in installed[: request.layer]:
                hidden = np.tanh(hidden @ matrix)
            path = request.output_dir / "routed-layer-inputs.npy"
            np.save(path, hidden)
            metadata |= {
                "layer": request.layer,
                "predecessor_state_hash": request.predecessor_state_hash,
                "capture_sha256": sha256_file(path),
            }
        elif kind == "causal_fit":
            capture_id = f"causal_fit_capture.layer_{request.layer:03d}"
            routed = np.load(_artifact_path(request, capture_id) / "routed-layer-inputs.npy")
            covariance = routed.T @ routed / routed.shape[0]
            path = request.output_dir / "causal-covariance.npy"
            np.save(path, covariance)
            metadata |= {
                "layer": request.layer,
                "predecessor_state_hash": request.predecessor_state_hash,
                "fit_sha256": sha256_file(path),
                "transient_files": [path.name],
            }
        elif kind == "causal_candidates":
            fit_id = f"causal_fit.layer_{request.layer:03d}"
            covariance = np.load(_artifact_path(request, fit_id) / "causal-covariance.npy")
            rows = []
            for bits in (2, 3, 4):
                reconstructed = _quantize(weights[request.layer], bits)
                residual = weights[request.layer] - reconstructed
                rows.append({"bits": bits, "damage": float(np.trace(residual.T @ covariance @ residual))})
            path = request.output_dir / "causal-candidates.json"
            write_json(path, {"layer": request.layer, "rows": rows})
            metadata |= {
                "layer": request.layer,
                "predecessor_state_hash": request.predecessor_state_hash,
                "candidate_ledger_sha256": sha256_file(path),
            }
        elif kind == "causal_encode":
            candidates_id = f"causal_candidates.layer_{request.layer:03d}"
            candidates = json.loads((_artifact_path(request, candidates_id) / "causal-candidates.json").read_text())["rows"]
            choice = min(candidates, key=lambda row: row["damage"])
            installed = self._encoded_weights(request)
            reconstructed = _quantize(weights[request.layer], int(choice["bits"]))
            installed[request.layer] = reconstructed
            write_json(
                request.output_dir / "encoded.json",
                {
                    "layer": request.layer,
                    "bits": choice["bits"],
                    "weights": reconstructed.tolist(),
                    "predecessor_state_hash": request.predecessor_state_hash,
                },
            )
            checkpoint_path = request.output_dir / "installed-checkpoint.npy"
            np.save(checkpoint_path, installed)
            metadata |= {
                "layer": request.layer,
                "predecessor_state_hash": request.predecessor_state_hash,
                "installed_checkpoint_sha256": sha256_file(checkpoint_path),
            }
        elif kind == "kld_reanchor":
            encoded = self._encoded_weights(request)
            hidden = features.copy()
            for matrix in encoded:
                hidden = np.tanh(hidden @ matrix)
            teacher_path = _artifact_path(request, "teacher_capture") / "teacher-logits.npy"
            reference = np.load(teacher_path)
            value = _kld(reference, hidden)
            threshold = value + 1e-9 if self.gate_passes else max(0.0, value - 1e-9)
            reference_path = request.output_dir / "reference-logits.npy"
            capture_path = request.output_dir / "student-logits.npy"
            np.save(reference_path, reference)
            np.save(capture_path, hidden)
            metadata["gate"] = {
                "passed": self.gate_passes,
                "metric": "kld",
                "value": value,
                "threshold": threshold,
                "reference_sha256": sha256_file(reference_path),
                "capture_sha256": sha256_file(capture_path),
            }
            metadata["gate_files"] = {
                "reference": reference_path.name,
                "capture": capture_path.name,
            }
        elif kind == "checkpoint_emission":
            encoded = self._encoded_weights(request)
            np.save(request.output_dir / "checkpoint.npy", encoded)
            metadata["checkpoint_sha256"] = sha256_file(request.output_dir / "checkpoint.npy")
        elif kind == "checkpoint_audit":
            checkpoint = _artifact_path(request, "checkpoint_emission") / "checkpoint.npy"
            value = np.load(checkpoint)
            if value.shape != weights.shape or not np.isfinite(value).all():
                raise ValueError("invalid tiny checkpoint")
            write_json(request.output_dir / "audit.json", {"ok": True, "checkpoint_sha256": sha256_file(checkpoint)})
            metadata["ok"] = True
        elif kind == "student_capture":
            checkpoint = np.load(_artifact_path(request, "checkpoint_emission") / "checkpoint.npy")
            hidden = features.copy()
            for matrix in checkpoint:
                hidden = np.tanh(hidden @ matrix)
            np.save(request.output_dir / "student-logits.npy", hidden)
            metadata["positions"] = int(hidden.shape[0])
        elif kind == "final_kld":
            teacher = np.load(_artifact_path(request, "teacher_capture") / "teacher-logits.npy")
            student = np.load(_artifact_path(request, "student_capture") / "student-logits.npy")
            value = _kld(teacher, student)
            write_json(request.output_dir / "final-kld.json", {"kld": value})
            metadata["kld"] = value
        else:  # pragma: no cover - catches runner/adapter contract expansion
            raise AssertionError(kind)
        return StageResult(metadata)


def _campaign_files(
    tmp_path: Path,
    *,
    layers=(0, 1, 2, 3, 4),
    interval=3,
    policy="continue",
    max_generations=None,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260823)
    source = tmp_path / "tiny-model.npz"
    np.savez(source, weights=rng.normal(0, 0.3, (len(layers), 4, 4)), features=rng.normal(0, 1, (12, 4)))
    corpus = tmp_path / "sealed-corpus.json"
    windows = {}
    for role_index, role in enumerate(("fit", "selection", "confirmation", "final")):
        tokens = [100 + role_index, 200 + role_index]
        windows[role] = [
            {
                "document_id": f"doc-{role}",
                "domain": "test",
                "offset": 0,
                "token_ids": tokens,
                "token_sha256": sha256_bytes(canonical_json(tokens)),
            }
        ]
    corpus_document = {
        "schema": "quant-pipeline.sealed-corpus.v1",
        "seed": 1,
        "window_tokens": 2,
        "minimum_domains": 1,
        "tokenizer": {"id": "tiny/tokenizer", "revision": REVISION},
        "source": {"path": "fixture", "sha256": sha256_bytes(b"fixture")},
        "role_counts": {role: 1 for role in windows},
        "windows": windows,
    }
    corpus_document["seal_sha256"] = sha256_bytes(canonical_json(corpus_document))
    write_json(corpus, corpus_document)
    kld_root = tmp_path / "kld-window"
    kld_root.mkdir()
    prefix = kld_root / "source-prefix.txt"
    prefix.write_text("tiny historical control")
    kld_tokens = list(range(2048))
    kld_document = {
        "schema": "quant-pipeline.kld-window.v1",
        "method": "glm-wikitext-2-raw-test-prefix-v1",
        "dataset": {"revision": "b" * 40},
        "construction": {},
        "source_prefix": {"file": prefix.name, "bytes": prefix.stat().st_size, "sha256": sha256_file(prefix)},
        "tokenizer": {"expected_model_revision": REVISION},
        "context_length": 2048,
        "prediction_positions": 2047,
        "token_ids": kld_tokens,
        "token_sha256": sha256_bytes(canonical_json(kld_tokens)),
        "first_16_token_ids": kld_tokens[:16],
    }
    kld_document["seal_sha256"] = sha256_bytes(canonical_json(kld_document))
    write_json(kld_root / "kld-window.json", kld_document)
    experiment = tmp_path / "experiment.toml"
    experiment.write_text(
        f'''name = "tiny-moe"
output_dir = "unused"

[model]
model_id = "tiny/moe"
revision = "{REVISION}"
family = "qwen3_moe"
local_path = "{source}"

[corpus]
input_jsonl = "{corpus}"
tokenizer_id = "tiny/tokenizer"
window_tokens = 32
fit_windows = 1
selection_windows = 1
confirmation_windows = 1
final_windows = 1
minimum_domains = 1

[objective]
reanchor_every_layers = {interval}
'''
    )
    definition = tmp_path / "campaign.json"
    definition_document = {
            "schema": "quant-pipeline.campaign-definition.v1",
            "experiment_spec": str(experiment),
            "inputs": {"source_checkpoint": str(source), "sealed_corpus": str(corpus), "kld_window": str(kld_root)},
            "layers": list(layers),
            "reanchor_every_layers": interval,
            "reanchor_failure_policy": policy,
            "retention_mode": "capture-plus-ledger",
    }
    if max_generations is not None:
        definition_document["max_generations"] = max_generations
    write_json(definition, definition_document)
    return definition, source


def test_tiny_moe_campaign_runs_causally_end_to_end(tmp_path):
    definition, _ = _campaign_files(tmp_path)
    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter()
    plan = create_plan(definition, campaign, adapter)
    assert status_campaign(campaign, adapter)["completed_stage_count"] == 0
    result = CampaignRunner(campaign, adapter).execute()
    assert result["complete"] is True
    assert result["integrity_ok"] is True
    assert result["completed_stage_count"] == len(plan["stages"])
    assert adapter.calls["kld_reanchor.block_000"] == 1
    assert adapter.calls["kld_reanchor.block_001"] == 1
    audit = audit_campaign(campaign, adapter)
    assert audit["integrity_ok"] is True
    events = [json.loads(line) for line in (campaign / "events.jsonl").read_text().splitlines()]
    encode_completions = [event for event in events if event["event"] == "stage_completed" and event["kind"] == "causal_encode"]
    assert all(row["predecessor_state_hash"] != "0" * 64 for row in encode_completions)
    assert len({row["details"]["installed_state_hash"] for row in encode_completions}) == 5
    retirements = [event for event in events if event["event"] == "artifacts_retirement_completed"]
    assert len(retirements) == 6  # one provisional fit plus one causal fit per layer
    fit_artifact = campaign / audit["completed"]["fit"]["path"]
    assert not (fit_artifact / "covariance.npy").exists()
    assert (fit_artifact / ".runner-result.json").is_file()
    final = next(row for row in audit["completed"] if row == "final_kld")
    assert final == "final_kld"


def test_post_freeze_confirmation_is_a_real_admission_gate(tmp_path):
    definition, _ = _campaign_files(tmp_path, layers=(0,), interval=1, policy="continue")
    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter(confirmation_passes=False)
    plan = create_plan(definition, campaign, adapter)
    stages = {row["stage_id"]: row for row in plan["stages"]}
    assert tuple(stages["confirmation"]["dependencies"]) == ("candidates", "allocation")
    assert "confirmation" in stages["causal_fit_capture.layer_000"]["dependencies"]

    result = CampaignRunner(campaign, adapter).execute()
    assert result["complete"] is False
    assert result["generation"] == 1
    assert result["next_stage"] == "allocation"
    assert "causal_fit_capture.layer_000" not in adapter.calls
    decision = next(
        event
        for event in EventJournal(campaign / "events.jsonl").read()
        if event["event"] == "gate_decision" and event["kind"] == "confirmation"
    )
    assert decision["details"]["gate"]["passed"] is False
    # A historical re-anchor policy of `continue` cannot bypass this gate.
    assert decision["details"]["action"] == "request_reallocation"


def test_every_causal_stage_binds_complete_five_layer_installed_prefix(tmp_path):
    definition, _ = _campaign_files(tmp_path, layers=(0, 1, 2, 3, 4), interval=3)
    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter()
    create_plan(definition, campaign, adapter)
    CampaignRunner(campaign, adapter).execute()
    events = [json.loads(line) for line in (campaign / "events.jsonl").read_text().splitlines()]
    starts = [event for event in events if event["event"] == "stage_started"]
    for event in starts:
        prefix = event["installed_layer_prefix"]
        layers = [row["layer"] for row in prefix]
        if event["kind"].startswith("causal_"):
            assert layers == list(range(int(event["stage_id"].rsplit("_", 1)[1])))
        for previous, current in zip(prefix, prefix[1:]):
            assert current["predecessor_state_hash"] == previous["installed_state_hash"]
        if prefix:
            assert prefix[-1]["installed_state_hash"] == event["predecessor_state_hash"]


def test_resume_reconstructs_gate_decision_without_rerunning_sealed_reanchor(tmp_path):
    definition, _ = _campaign_files(tmp_path, layers=(0, 1), interval=2)
    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter()
    create_plan(definition, campaign, adapter)
    runner = CampaignRunner(campaign, adapter)
    original_append = runner.journal.append
    crashed = False

    def crash_before_gate(body):
        nonlocal crashed
        if (
            not crashed
            and body.get("event") == "gate_decision"
            and body.get("kind") == "kld_reanchor"
        ):
            crashed = True
            raise KeyboardInterrupt("after sealed reanchor")
        return original_append(body)

    runner.journal.append = crash_before_gate
    with pytest.raises(KeyboardInterrupt, match="after sealed reanchor"):
        runner.execute()
    assert adapter.calls["kld_reanchor.block_000"] == 1
    pending = audit_campaign(campaign, adapter)
    assert pending["integrity_ok"] is True
    assert pending["pending_gate"] == "kld_reanchor.block_000"
    resumed = CampaignRunner(campaign, adapter).execute(resume=True)
    assert resumed["complete"] is True
    assert adapter.calls["kld_reanchor.block_000"] == 1


@pytest.mark.parametrize("policy", ["rollback", "request_reallocation"])
def test_resume_completes_gate_supersession_crash_window_exactly_once(tmp_path, policy):
    definition, _ = _campaign_files(
        tmp_path,
        layers=(0, 1),
        interval=2,
        policy=policy,
    )
    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter(gate_passes=False)
    create_plan(definition, campaign, adapter)
    runner = CampaignRunner(campaign, adapter)
    original_append = runner.journal.append
    crashed = False

    def crash_before_supersession(body):
        nonlocal crashed
        if not crashed and body.get("event") == "generation_superseded":
            crashed = True
            raise KeyboardInterrupt("after gate decision before supersession")
        return original_append(body)

    runner.journal.append = crash_before_supersession
    with pytest.raises(KeyboardInterrupt, match="before supersession"):
        runner.execute()
    pending = audit_campaign(campaign, adapter)
    assert pending["integrity_ok"] is True
    assert pending["pending_gate"] is None
    assert pending["pending_supersession"] == "kld_reanchor.block_000"
    assert pending["disposition"] == "replan_pending"
    assert adapter.calls["kld_reanchor.block_000"] == 1

    resumed = CampaignRunner(campaign, adapter).execute(resume=True)
    assert resumed["integrity_ok"] is True
    assert resumed["generation"] == 1
    assert resumed["disposition"] == "replan_ready"
    assert resumed["next_stage"] == "allocation"
    assert resumed["pending_supersession"] is None
    assert adapter.calls["kld_reanchor.block_000"] == 1
    supersessions = [
        event
        for event in EventJournal(campaign / "events.jsonl").read()
        if event["event"] == "generation_superseded"
    ]
    assert len(supersessions) == 1


def test_failed_gate_continue_policy_advances_without_supersession(tmp_path):
    definition, _ = _campaign_files(
        tmp_path,
        layers=(0, 1),
        interval=2,
        policy="continue",
    )
    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter(gate_passes=False)
    create_plan(definition, campaign, adapter)
    result = CampaignRunner(campaign, adapter).execute()
    assert result["integrity_ok"] is True
    assert result["complete"] is True
    assert result["generation"] == 0
    assert result["pending_supersession"] is None
    assert not any(
        event["event"] == "generation_superseded"
        for event in EventJournal(campaign / "events.jsonl").read()
    )


def test_torn_gate_decision_tail_recovers_and_reconstructs_gate_once(tmp_path):
    definition, _ = _campaign_files(tmp_path, layers=(0, 1), interval=2)
    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter()
    create_plan(definition, campaign, adapter)
    runner = CampaignRunner(campaign, adapter)
    original_append = runner.journal.append
    crashed = False

    def tear_gate_decision(body):
        nonlocal crashed
        if (
            not crashed
            and body.get("event") == "gate_decision"
            and body.get("kind") == "kld_reanchor"
        ):
            crashed = True
            with (campaign / "events.jsonl").open("ab") as handle:
                handle.write(b'{"event":"gate_decision"')
                handle.flush()
                os.fsync(handle.fileno())
            raise KeyboardInterrupt("torn gate decision")
        return original_append(body)

    runner.journal.append = tear_gate_decision
    with pytest.raises(KeyboardInterrupt, match="torn gate decision"):
        runner.execute()
    resumed = CampaignRunner(campaign, adapter).execute(resume=True)
    assert resumed["integrity_ok"] is True
    assert resumed["complete"] is True
    assert resumed["pending_gate"] is None
    assert adapter.calls["kld_reanchor.block_000"] == 1
    recovery_events = [
        event
        for event in EventJournal(campaign / "events.jsonl").read()
        if event["event"] == "journal_tail_recovered"
    ]
    assert len(recovery_events) == 1


@pytest.mark.parametrize("policy", ["rollback", "request_reallocation"])
def test_torn_gate_tail_recovers_noncontinue_decision_and_supersession(tmp_path, policy):
    definition, _ = _campaign_files(
        tmp_path, layers=(0, 1), interval=2, policy=policy
    )
    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter(gate_passes=False)
    create_plan(definition, campaign, adapter)
    runner = CampaignRunner(campaign, adapter)
    original_append = runner.journal.append
    torn = False

    def tear_gate_decision(body):
        nonlocal torn
        if (
            not torn
            and body.get("event") == "gate_decision"
            and body.get("kind") == "kld_reanchor"
        ):
            torn = True
            with (campaign / "events.jsonl").open("ab") as handle:
                handle.write(b'{"event":"gate_decision"')
                handle.flush()
                os.fsync(handle.fileno())
            raise KeyboardInterrupt("torn noncontinue gate decision")
        return original_append(body)

    runner.journal.append = tear_gate_decision
    with pytest.raises(KeyboardInterrupt, match="torn noncontinue"):
        runner.execute()
    resumed = CampaignRunner(campaign, adapter).execute(resume=True)
    assert resumed["integrity_ok"] is True
    assert resumed["generation"] == 1
    assert resumed["pending_gate"] is None
    assert resumed["pending_supersession"] is None
    events = EventJournal(campaign / "events.jsonl").read()
    assert [event["event"] for event in events].count("journal_tail_recovered") == 1
    assert sum(
        event["event"] == "gate_decision" and event.get("kind") == "kld_reanchor"
        for event in events
    ) == 1
    assert [event["event"] for event in events].count("generation_superseded") == 1


@pytest.mark.parametrize("policy", ["rollback", "request_reallocation"])
def test_torn_supersession_tail_recovers_after_gate_exactly_once(tmp_path, policy):
    definition, _ = _campaign_files(
        tmp_path, layers=(0, 1), interval=2, policy=policy
    )
    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter(gate_passes=False)
    create_plan(definition, campaign, adapter)
    runner = CampaignRunner(campaign, adapter)
    original_append = runner.journal.append
    torn = False

    def tear_supersession(body):
        nonlocal torn
        if not torn and body.get("event") == "generation_superseded":
            torn = True
            with (campaign / "events.jsonl").open("ab") as handle:
                handle.write(b'{"event":"generation_superseded"')
                handle.flush()
                os.fsync(handle.fileno())
            raise KeyboardInterrupt("torn generation supersession")
        return original_append(body)

    runner.journal.append = tear_supersession
    with pytest.raises(KeyboardInterrupt, match="torn generation"):
        runner.execute()
    resumed = CampaignRunner(campaign, adapter).execute(resume=True)
    assert resumed["integrity_ok"] is True
    assert resumed["generation"] == 1
    assert resumed["pending_supersession"] is None
    events = EventJournal(campaign / "events.jsonl").read()
    assert [event["event"] for event in events].count("journal_tail_recovered") == 1
    assert sum(
        event["event"] == "gate_decision" and event.get("kind") == "kld_reanchor"
        for event in events
    ) == 1
    assert [event["event"] for event in events].count("generation_superseded") == 1


@pytest.mark.parametrize(
    ("mutation", "failure"),
    [
        ("stale_generation", "invalid-generation-supersession"),
        ("predecessor", "generation-gate-binding-mismatch"),
        ("decision_hash", "generation-gate-binding-mismatch"),
        ("action", "generation-gate-binding-mismatch"),
        ("generation_context", "generation-gate-binding-mismatch"),
        ("invalidation_extra", "generation-invalidation-record-malformed"),
    ],
)
def test_supersession_replay_rejects_stale_or_mismatched_bindings(tmp_path, mutation, failure):
    definition, _ = _campaign_files(
        tmp_path, layers=(0, 1), interval=2, policy="rollback"
    )
    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter(gate_passes=False)
    create_plan(definition, campaign, adapter)
    runner = CampaignRunner(campaign, adapter)
    original_append = runner.journal.append

    def tamper_supersession(body):
        if body.get("event") != "generation_superseded":
            return original_append(body)
        altered = copy.deepcopy(body)
        if mutation == "stale_generation":
            altered["details"]["prior_generation"] = 7
        elif mutation == "predecessor":
            altered["predecessor_state_hash"] = "f" * 64
        elif mutation == "decision_hash":
            altered["input_hashes"]["gate_decision_sha256"] = "f" * 64
        elif mutation == "generation_context":
            altered["generation_context"] = {"unbound": True}
        elif mutation == "invalidation_extra":
            altered["details"]["invalidated_artifacts"][0]["unbound"] = True
        else:
            altered["details"]["action"] = "request_reallocation"
        return original_append(altered)

    runner.journal.append = tamper_supersession
    result = runner.execute()
    assert result["integrity_ok"] is False
    assert any(failure in item for item in result["failures"])


def test_replay_rejects_duplicate_generation_supersession(tmp_path):
    definition, _ = _campaign_files(
        tmp_path, layers=(0, 1), interval=2, policy="rollback"
    )
    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter(gate_passes=False)
    create_plan(definition, campaign, adapter)
    result = CampaignRunner(campaign, adapter).execute()
    assert result["generation"] == 1
    journal = EventJournal(campaign / "events.jsonl")
    supersession = next(
        event for event in journal.read() if event["event"] == "generation_superseded"
    )
    chain_fields = {
        "schema", "event_index", "timestamp", "previous_event_sha256", "event_sha256"
    }
    journal.append({key: value for key, value in supersession.items() if key not in chain_fields})
    damaged = audit_campaign(campaign, adapter)
    assert damaged["integrity_ok"] is False
    assert any("invalid-generation-supersession" in item for item in damaged["failures"])


def test_gate_replay_binds_installed_prefix_and_request_fields(tmp_path):
    definition, _ = _campaign_files(
        tmp_path, layers=(0, 1), interval=2, policy="continue"
    )
    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter()
    create_plan(definition, campaign, adapter)
    runner = CampaignRunner(campaign, adapter)
    original_append = runner.journal.append

    def tamper_gate(body):
        if body.get("event") != "gate_decision":
            return original_append(body)
        altered = copy.deepcopy(body)
        altered["installed_layer_prefix"] = []
        return original_append(altered)

    runner.journal.append = tamper_gate
    result = runner.execute()
    assert result["integrity_ok"] is False
    assert any("gate-installed_layer_prefix-mismatch" in item for item in result["failures"])


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_plan_rejects_invalid_generation_cap(tmp_path, invalid):
    definition, _ = _campaign_files(
        tmp_path, layers=(0,), interval=1, max_generations=invalid
    )
    with pytest.raises(ValueError, match="max_generations"):
        create_plan(definition, tmp_path / "campaign", TinyMoeAdapter())


def test_generation_cap_allows_boundary_then_fails_closed_without_history_deletion(tmp_path):
    definition, _ = _campaign_files(
        tmp_path,
        layers=(0, 1),
        interval=2,
        policy="request_reallocation",
        max_generations=2,
    )
    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter(gate_passes=False)
    create_plan(definition, campaign, adapter)
    first = CampaignRunner(campaign, adapter).execute()
    assert first["generation"] == 1
    assert first["integrity_ok"] is True

    with pytest.raises(RuntimeError, match="generation cap reached"):
        CampaignRunner(campaign, adapter).execute(resume=True)
    capped = audit_campaign(campaign, adapter)
    assert capped["integrity_ok"] is True
    assert capped["generation"] == 1
    assert capped["pending_supersession"] == "kld_reanchor.block_000"
    events_before = EventJournal(campaign / "events.jsonl").read()
    assert [event["event"] for event in events_before].count("generation_superseded") == 1
    assert sum(
        event["event"] == "gate_decision" and event.get("kind") == "kld_reanchor"
        for event in events_before
    ) == 2

    with pytest.raises(RuntimeError, match="generation cap reached"):
        CampaignRunner(campaign, adapter).execute(resume=True)
    events_after = EventJournal(campaign / "events.jsonl").read()
    assert events_after == events_before

    decision = next(
        event
        for event in reversed(events_after)
        if event["event"] == "gate_decision" and event["generation"] == 1
    )
    completed = capped["completed"]
    plan = json.loads((campaign / "plan.json").read_text())
    stage_order = [row["stage_id"] for row in plan["stages"]]
    invalidated = [
        {
            "stage_id": stage_id,
            "path": completed[stage_id]["path"],
            "artifact_sha256": completed[stage_id]["artifact_sha256"],
        }
        for stage_id in stage_order[stage_order.index("allocation") :]
        if stage_id in completed
    ]
    failed_gate = decision["stage_id"]
    failed_hash = completed[failed_gate]["artifact_sha256"]
    rollback = decision["details"]["rollback_state_hash"]
    EventJournal(campaign / "events.jsonl").append(
        {
            "event": "generation_superseded",
            "plan_sha256": plan["plan_sha256"],
            "stage_id": None,
            "kind": "generation_replan",
            "attempt": 2,
            "generation": 1,
            "generation_context": decision["generation_context"],
            "input_hashes": {
                "failed_gate_artifact_sha256": failed_hash,
                "gate_decision_sha256": decision["event_sha256"],
            },
            "output_hashes": {},
            "predecessor_state_hash": rollback,
            "details": {
                "prior_generation": 1,
                "new_generation": 2,
                "failed_gate_stage_id": failed_gate,
                "failed_gate_artifact_sha256": failed_hash,
                "gate_decision_sha256": decision["event_sha256"],
                "action": decision["details"]["action"],
                "rollback_state_hash": rollback,
                "invalidated_artifacts": invalidated,
            },
        }
    )
    forged = audit_campaign(campaign, adapter)
    assert forged["integrity_ok"] is False
    assert any("invalid-generation-supersession" in item for item in forged["failures"])


def test_single_generation_cap_accepts_continue_policy_without_supersession(tmp_path):
    definition, _ = _campaign_files(
        tmp_path,
        layers=(0,),
        interval=1,
        policy="continue",
        max_generations=1,
    )
    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter(gate_passes=False)
    create_plan(definition, campaign, adapter)
    result = CampaignRunner(campaign, adapter).execute()
    assert result["complete"] is True
    assert result["generation"] == 0


def test_interrupted_campaign_resumes_without_rerunning_completed_stages(tmp_path):
    definition, _ = _campaign_files(tmp_path, layers=(0, 1), interval=2)
    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter(fail_once="candidates")
    create_plan(definition, campaign, adapter)
    with pytest.raises(RuntimeError, match="injected interruption"):
        CampaignRunner(campaign, adapter).execute()
    calls_before = dict(adapter.calls)
    result = CampaignRunner(campaign, adapter).execute(resume=True)
    assert result["complete"] is True
    assert adapter.calls["identity"] == calls_before["identity"] == 1
    assert adapter.calls["fit"] == calls_before["fit"] == 1
    assert adapter.calls["candidates"] == 2
    with pytest.raises(RuntimeError, match="use resume"):
        CampaignRunner(campaign, adapter).execute()


def test_hard_interruption_after_result_receipt_recovers_without_rerun(tmp_path):
    definition, _ = _campaign_files(tmp_path, layers=(0,), interval=1)
    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter()
    create_plan(definition, campaign, adapter)
    runner = CampaignRunner(campaign, adapter)
    original_append = runner.journal.append
    interrupted = False

    def crash_before_completion(body):
        nonlocal interrupted
        if not interrupted and body.get("event") == "stage_completed" and body.get("stage_id") == "identity":
            interrupted = True
            raise KeyboardInterrupt("simulated hard exit")
        return original_append(body)

    runner.journal.append = crash_before_completion
    with pytest.raises(KeyboardInterrupt, match="simulated hard exit"):
        runner.execute()
    assert adapter.calls["identity"] == 1
    assert (campaign / "attempts" / "identity" / "attempt-0001" / ".runner-result.json").is_file()
    resumed = CampaignRunner(campaign, adapter).execute(resume=True)
    assert resumed["complete"] is True
    assert adapter.calls["identity"] == 1


def test_hard_interruption_rejects_payload_mutated_after_receipt(tmp_path):
    definition, _ = _campaign_files(tmp_path, layers=(0,), interval=1)
    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter()
    create_plan(definition, campaign, adapter)
    runner = CampaignRunner(campaign, adapter)
    original_append = runner.journal.append
    interrupted = False

    def crash_before_completion(body):
        nonlocal interrupted
        if not interrupted and body.get("event") == "stage_completed" and body.get("stage_id") == "identity":
            interrupted = True
            raise KeyboardInterrupt("simulated hard exit")
        return original_append(body)

    runner.journal.append = crash_before_completion
    with pytest.raises(KeyboardInterrupt):
        runner.execute()
    identity = campaign / "attempts" / "identity" / "attempt-0001" / "identity.json"
    identity.write_text('{"corrupted":true}\n')
    with pytest.raises(ValueError, match="payload bytes differ"):
        CampaignRunner(campaign, adapter).execute(resume=True)


def test_resume_recovers_only_a_torn_terminal_journal_fragment(tmp_path):
    definition, _ = _campaign_files(tmp_path, layers=(0,), interval=1)
    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter()
    create_plan(definition, campaign, adapter)
    with (campaign / "events.jsonl").open("ab") as handle:
        handle.write(b'{"schema":"quant-pipeline.campaign-event.v1"')
    assert audit_campaign(campaign, adapter)["integrity_ok"] is False
    result = CampaignRunner(campaign, adapter).execute(resume=True)
    assert result["complete"] is True
    assert result["integrity_ok"] is True
    events = [json.loads(line) for line in (campaign / "events.jsonl").read_text().splitlines()]
    recovered = [event for event in events if event["event"] == "journal_tail_recovered"]
    assert len(recovered) == 1
    quarantine = campaign / recovered[0]["details"]["quarantine_path"]
    assert quarantine.is_file()
    assert sha256_file(quarantine) == recovered[0]["output_hashes"]["fragment_sha256"]


def test_resume_reconciles_retirement_after_completion_crash_window(tmp_path):
    definition, _ = _campaign_files(tmp_path, layers=(0,), interval=1)
    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter()
    create_plan(definition, campaign, adapter)
    runner = CampaignRunner(campaign, adapter)
    original = runner._plan_retirement_after
    interrupted = False

    def crash_after_candidate_completion(stage, completed):
        nonlocal interrupted
        if not interrupted and stage.kind == "candidates":
            interrupted = True
            raise KeyboardInterrupt("retirement planning crash")
        return original(stage, completed)

    runner._plan_retirement_after = crash_after_candidate_completion
    with pytest.raises(KeyboardInterrupt, match="retirement planning crash"):
        runner.execute()
    fit = audit_campaign(campaign, adapter)["completed"]["fit"]
    fit_root = campaign / fit["path"]
    assert (fit_root / "covariance.npy").is_file()
    result = CampaignRunner(campaign, adapter).execute(resume=True)
    assert result["complete"] is True
    assert not (fit_root / "covariance.npy").exists()
    events = [json.loads(line) for line in (campaign / "events.jsonl").read_text().splitlines()]
    assert any(event["event"] == "artifacts_retirement_completed" for event in events)


def test_resume_reconciles_retirement_against_exact_cross_generation_artifacts(tmp_path):
    class GenerationGateAdapter(TinyMoeAdapter):
        def identity(self):
            return {"name": "generation-gate", "version": 1}

        def run(self, request):
            self.gate_passes = request.generation >= 1
            return super().run(request)

    definition, _ = _campaign_files(
        tmp_path, layers=(0, 1), interval=2, policy="rollback"
    )
    campaign = tmp_path / "campaign"
    adapter = GenerationGateAdapter()
    create_plan(definition, campaign, adapter)
    CampaignRunner(campaign, adapter).execute()

    runner = CampaignRunner(campaign, adapter)
    original = runner._plan_retirement_after
    interrupted = False

    def crash_before_generation_one_retirement(stage, completed):
        nonlocal interrupted
        if not interrupted and stage.kind == "causal_candidates":
            interrupted = True
            raise KeyboardInterrupt("cross-generation retirement crash")
        return original(stage, completed)

    runner._plan_retirement_after = crash_before_generation_one_retirement
    with pytest.raises(KeyboardInterrupt, match="cross-generation retirement crash"):
        runner.execute(resume=True)

    result = CampaignRunner(campaign, adapter).execute(resume=True)
    audit = audit_campaign(campaign, adapter)
    current_fit = audit["completed"]["causal_fit.layer_000"]
    current_covariance = campaign / current_fit["path"] / "causal-covariance.npy"
    plans = [
        event
        for event in EventJournal(campaign / "events.jsonl").read()
        if event["event"] == "artifacts_retirement_planned"
        and event.get("input_hashes", {}).get("producer_artifact_sha256")
        == current_fit["artifact_sha256"]
    ]
    assert result["complete"] is True
    assert audit["integrity_ok"] is True
    assert len(plans) == 1
    assert not current_covariance.exists()


@pytest.mark.parametrize(
    "output_hashes",
    [
        None,
        {"artifact": {}},
        {"artifact": {"path": 123}},
        {"artifact": {"path": "../outside"}},
    ],
)
def test_audit_fails_closed_for_malformed_completed_artifact_shapes(tmp_path, output_hashes):
    definition, _ = _campaign_files(tmp_path, layers=(0,), interval=1)
    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter(fail_once="candidates")
    create_plan(definition, campaign, adapter)
    with pytest.raises(RuntimeError, match="injected interruption"):
        CampaignRunner(campaign, adapter).execute()

    journal = EventJournal(campaign / "events.jsonl")
    failed = next(
        event
        for event in journal.read()
        if event["event"] == "stage_failed" and event["stage_id"] == "candidates"
    )
    chain_fields = {
        "schema", "event_index", "timestamp", "previous_event_sha256", "event_sha256"
    }
    forged = {key: value for key, value in failed.items() if key not in chain_fields}
    forged["event"] = "stage_completed"
    forged["output_hashes"] = output_hashes
    forged["details"] = {"metadata": {}, "installed_state_hash": failed["predecessor_state_hash"]}
    journal.append(forged)

    audit = audit_campaign(campaign, adapter)
    assert audit["integrity_ok"] is False
    assert audit["failures"]


def test_replay_rejects_duplicate_gate_decision_per_generation(tmp_path):
    definition, _ = _campaign_files(tmp_path, layers=(0, 1), interval=2)
    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter()
    create_plan(definition, campaign, adapter)
    CampaignRunner(campaign, adapter).execute()

    journal = EventJournal(campaign / "events.jsonl")
    decision = next(event for event in journal.read() if event["event"] == "gate_decision")
    chain_fields = {
        "schema", "event_index", "timestamp", "previous_event_sha256", "event_sha256"
    }
    journal.append({key: value for key, value in decision.items() if key not in chain_fields})

    audit = audit_campaign(campaign, adapter)
    assert audit["integrity_ok"] is False
    assert any("duplicate-gate-decision" in failure for failure in audit["failures"])


def test_audit_rejects_completed_artifact_root_symlink_outside_campaign(tmp_path):
    definition, _ = _campaign_files(tmp_path, layers=(0,), interval=1)
    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter()
    create_plan(definition, campaign, adapter)
    CampaignRunner(campaign, adapter).execute()

    identity = audit_campaign(campaign, adapter)["completed"]["identity"]
    artifact_root = campaign / identity["path"]
    external = tmp_path / "external-identity"
    artifact_root.rename(external)
    artifact_root.symlink_to(external, target_is_directory=True)

    audit = audit_campaign(campaign, adapter)
    assert audit["integrity_ok"] is False
    assert any("artifact:identity:ValueError" in failure for failure in audit["failures"])


def test_pending_retirement_never_unlinks_through_symlinked_producer(tmp_path):
    definition, _ = _campaign_files(tmp_path, layers=(0,), interval=1)
    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter()
    create_plan(definition, campaign, adapter)
    runner = CampaignRunner(campaign, adapter)
    original = runner._finalize_retirement
    interrupted = False

    def crash_before_unlink(planned):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt("before retirement unlink")
        return original(planned)

    runner._finalize_retirement = crash_before_unlink
    with pytest.raises(KeyboardInterrupt, match="before retirement unlink"):
        runner.execute()

    fit = audit_campaign(campaign, adapter)["completed"]["fit"]
    producer_root = campaign / fit["path"]
    external = tmp_path / "external-fit"
    producer_root.rename(external)
    producer_root.symlink_to(external, target_is_directory=True)
    external_covariance = external / "covariance.npy"
    assert external_covariance.is_file()

    with pytest.raises(ValueError, match="campaign integrity failure"):
        CampaignRunner(campaign, adapter).execute(resume=True)
    assert external_covariance.is_file()


@pytest.mark.parametrize(
    ("field", "value", "failure"),
    [
        ("generation", [], "generation-binding-mismatch"),
        ("stage_id", {"unhashable": True}, "invalid-stage-id"),
    ],
)
def test_audit_fails_closed_for_unhashable_stage_identity_fields(
    tmp_path, field, value, failure
):
    definition, _ = _campaign_files(tmp_path, layers=(0,), interval=1)
    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter()
    create_plan(definition, campaign, adapter)
    CampaignRunner(campaign, adapter).execute()
    journal = EventJournal(campaign / "events.jsonl")
    decision = next(event for event in journal.read() if event["event"] == "gate_decision")
    chain_fields = {
        "schema", "event_index", "timestamp", "previous_event_sha256", "event_sha256"
    }
    forged = {key: value for key, value in decision.items() if key not in chain_fields}
    forged[field] = value
    journal.append(forged)

    audit = audit_campaign(campaign, adapter)
    assert audit["integrity_ok"] is False
    assert any(failure in item for item in audit["failures"])


def test_retirement_completion_binds_exact_count_and_attempt(tmp_path):
    definition, _ = _campaign_files(tmp_path, layers=(0,), interval=1)
    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter()
    create_plan(definition, campaign, adapter)
    runner = CampaignRunner(campaign, adapter)
    original_append = runner.journal.append
    tampered = False

    def tamper_retirement_completion(body):
        nonlocal tampered
        if body.get("event") != "artifacts_retirement_completed" or tampered:
            return original_append(body)
        tampered = True
        altered = copy.deepcopy(body)
        altered["output_hashes"] = {"retired_file_count": 999}
        altered["attempt"] = 2
        return original_append(altered)

    runner.journal.append = tamper_retirement_completion
    result = runner.execute()
    assert result["integrity_ok"] is False
    assert any("retirement-completion-mismatch" in item for item in result["failures"])


def test_retirement_completion_is_unique_per_plan(tmp_path):
    definition, _ = _campaign_files(tmp_path, layers=(0,), interval=1)
    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter()
    create_plan(definition, campaign, adapter)
    CampaignRunner(campaign, adapter).execute()
    journal = EventJournal(campaign / "events.jsonl")
    completion = next(
        event for event in journal.read() if event["event"] == "artifacts_retirement_completed"
    )
    chain_fields = {
        "schema", "event_index", "timestamp", "previous_event_sha256", "event_sha256"
    }
    journal.append({key: value for key, value in completion.items() if key not in chain_fields})

    audit = audit_campaign(campaign, adapter)
    assert audit["integrity_ok"] is False
    assert any("duplicate-retirement-completion" in item for item in audit["failures"])


def test_kld_gate_rejects_contradictory_pass_flag(tmp_path):
    class ContradictoryGateAdapter(TinyMoeAdapter):
        def run(self, request):
            result = super().run(request)
            if request.kind == "kld_reanchor":
                result.metadata["gate"]["passed"] = not result.metadata["gate"]["passed"]
            return result

    definition, _ = _campaign_files(tmp_path, layers=(0,), interval=1)
    campaign = tmp_path / "campaign"
    adapter = ContradictoryGateAdapter()
    create_plan(definition, campaign, adapter)
    with pytest.raises(ValueError, match="contradicts"):
        CampaignRunner(campaign, adapter).execute()


def test_input_and_completed_artifact_drift_fail_closed(tmp_path):
    definition, source = _campaign_files(tmp_path, layers=(0,), interval=1)
    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter()
    create_plan(definition, campaign, adapter)
    with source.open("ab") as handle:
        handle.write(b"drift")
    status = status_campaign(campaign, adapter)
    assert "input:source_checkpoint:identity-drift" in status["drift"]
    with pytest.raises(ValueError, match="identity drift"):
        CampaignRunner(campaign, adapter).execute()

    clean_definition, _ = _campaign_files(tmp_path / "clean", layers=(0,), interval=1)
    clean_campaign = tmp_path / "clean-campaign"
    clean_adapter = TinyMoeAdapter()
    create_plan(clean_definition, clean_campaign, clean_adapter)
    CampaignRunner(clean_campaign, clean_adapter).execute()
    identity_artifact = audit_campaign(clean_campaign, clean_adapter)["completed"]["identity"]
    (clean_campaign / identity_artifact["path"] / "identity.json").write_text("{}")
    damaged = audit_campaign(clean_campaign, clean_adapter)
    assert damaged["integrity_ok"] is False
    assert any("artifact:identity:identity-drift" == row for row in damaged["failures"])


def test_failed_reanchor_rollback_stops_before_checkpoint(tmp_path):
    definition, _ = _campaign_files(tmp_path, layers=(0, 1), interval=2, policy="rollback")
    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter(gate_passes=False)
    create_plan(definition, campaign, adapter)
    result = CampaignRunner(campaign, adapter).execute()
    assert result["complete"] is False
    assert result["disposition"] == "replan_ready"
    assert result["generation"] == 1
    assert result["next_stage"] == "allocation"
    assert "checkpoint_emission" not in adapter.calls
    resumed = CampaignRunner(campaign, adapter).execute(resume=True)
    assert resumed["disposition"] == "replan_ready"
    assert resumed["generation"] == 2
    assert adapter.calls["allocation"] == 2
    assert "checkpoint_emission" not in adapter.calls


@pytest.mark.parametrize("interval", [0, 5])
def test_plan_rejects_invalid_reanchor_interval(tmp_path, interval):
    # Write the malformed definition directly because the experiment validator allows
    # legacy values while the causal runner contract must cap the interval at four.
    definition, _ = _campaign_files(tmp_path, layers=(0,), interval=interval)
    with pytest.raises(ValueError, match="reanchor_every_layers"):
        create_plan(definition, tmp_path / "campaign", TinyMoeAdapter())


@pytest.mark.parametrize("interval", [1, 4])
def test_plan_accepts_reanchor_interval_boundaries(tmp_path, interval):
    definition, _ = _campaign_files(
        tmp_path, layers=tuple(range(interval)), interval=interval
    )
    plan = create_plan(definition, tmp_path / "campaign", TinyMoeAdapter())
    assert plan["definition"]["reanchor_every_layers"] == interval


def test_plan_requires_empty_destination_and_journal_tamper_is_detected(tmp_path):
    definition, _ = _campaign_files(tmp_path, layers=(0,), interval=1)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "existing").write_text("do not overwrite")
    with pytest.raises(FileExistsError):
        create_plan(definition, occupied, TinyMoeAdapter())

    campaign = tmp_path / "campaign"
    adapter = TinyMoeAdapter()
    create_plan(definition, campaign, adapter)
    lines = (campaign / "events.jsonl").read_text().splitlines()
    event = json.loads(lines[0])
    event["details"]["stage_count"] += 1
    lines[0] = canonical_json(event).decode().rstrip("\n")
    (campaign / "events.jsonl").write_text("\n".join(lines) + "\n")
    damaged = audit_campaign(campaign, adapter)
    assert damaged["integrity_ok"] is False
    assert damaged["failures"][0].startswith("journal:ValueError:journal event hash mismatch")


def test_plan_and_status_do_not_preflight_or_execute_b200_work(tmp_path):
    class GuardedB200Adapter(TinyMoeAdapter):
        def __init__(self):
            super().__init__()
            self.preflight_calls = 0

        def preflight(self, plan):
            self.preflight_calls += 1
            return {
                "ok": False,
                "local_only": True,
                "remote_endpoints": [],
                "gpu": {"required": True, "minimum_count": 2, "minimum_capability": [10, 0]},
                "storage": {
                    "retention_mode": plan["definition"]["retention_mode"],
                    "estimated_peak_bytes": 10**12,
                    "available_bytes": 0,
                    "safety_margin_bytes": 10**9,
                },
                "software": {"cuda": "required"},
            }

    definition, _ = _campaign_files(tmp_path, layers=(0,), interval=1)
    campaign = tmp_path / "campaign"
    adapter = GuardedB200Adapter()
    create_plan(definition, campaign, adapter)
    status_campaign(campaign, adapter)
    assert adapter.preflight_calls == 0
    assert adapter.calls == {}
    with pytest.raises(RuntimeError, match="resource preflight"):
        CampaignRunner(campaign, adapter).execute()
    assert adapter.preflight_calls == 1
    assert adapter.calls == {}


def test_remote_endpoint_preflight_is_rejected_before_any_stage(tmp_path):
    class RemoteAdapter(TinyMoeAdapter):
        def preflight(self, plan):
            result = dict(super().preflight(plan))
            result["local_only"] = False
            result["remote_endpoints"] = ["ssh://forbidden.example"]
            return result

    definition, _ = _campaign_files(tmp_path, layers=(0,), interval=1)
    campaign = tmp_path / "campaign"
    adapter = RemoteAdapter()
    create_plan(definition, campaign, adapter)
    with pytest.raises(ValueError, match="local-only"):
        CampaignRunner(campaign, adapter).execute()
    assert adapter.calls == {}
