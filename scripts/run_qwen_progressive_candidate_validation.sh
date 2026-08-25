#!/usr/bin/env bash
set -euo pipefail

# Publish and validate one complete progressive-state Qwen candidate factory.
# The source causal allocation supplies only the K3/K4 rate choices. Candidate
# payload bytes, reconstruction hashes, and student logits come from the
# progressive factory and are independently bound through an immutable Hub
# inventory before the untouched 10x2048 SDPA panel is evaluated.

RUN_ROOT=${RUN_ROOT:-/artifacts/shapleymcg/qwen3-30b-a3b-v1/progressive-candidate-v1}
BASE_ROOT=${BASE_ROOT:-/qwen-shapleymcg-run}
SCRIPT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CODE_ROOT=${CODE_ROOT:-${SCRIPT_ROOT}}
PYTHON=${PYTHON:-/workspace/quant-venv/bin/python}
MODEL=${MODEL:-/models/Qwen3-30B-A3B-Base}
MODEL_REVISION=${MODEL_REVISION:-1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9}
SOURCE_ALLOCATION=${SOURCE_ALLOCATION:-/artifacts/shapleymcg/qwen3-30b-a3b-v1/causal-arm-v3/allocation-stage/allocation.json}
BASELINE_INVENTORY=${BASELINE_INVENTORY:-/artifacts/shapleymcg/qwen3-30b-a3b-v1/causal-arm-v3/candidate-inventory.json}
TEACHER_PANEL_ROOT=${TEACHER_PANEL_ROOT:-/artifacts/shapleymcg/qwen3-30b-a3b-v1/causal-arm-v3/turboderp-wiki2-sdpa-teacher}
HF_REPO=${HF_REPO:-brandonmusic/shapleymcg-qwen3-30b-a3b-reproducibility}
HF_PATH_PREFIX=${HF_PATH_PREFIX:-candidate-factories/progressive-state-v1}
HF_RESULT_PREFIX=${HF_RESULT_PREFIX:-results/qwen3-30b-a3b-base/progressive-candidate-v1}
HF_TOKEN_SOURCE=${HF_TOKEN_SOURCE:-/root/.cache/huggingface/token}
DRIVER_PID_FILE=${DRIVER_PID_FILE:-${RUN_ROOT}/logs/progressive-pipeline.pid}
VALIDATION_ROOT=${VALIDATION_ROOT:-${RUN_ROOT}/frozen-causal-rate-validation-v1}
CONTROL_BUNDLE=${CONTROL_BUNDLE:-/artifacts/shapleymcg/qwen3-30b-a3b-v1/control-bundle-v1}
BASE_CONTROL_REVISION=${BASE_CONTROL_REVISION:-c9c2b001dd943b8251fc0102ec76ab1b8d572219}
BASE_CONTROL_PREFIX=${BASE_CONTROL_PREFIX:-controls/fixed-hadamard-k34-v1}
GLM_MODEL_REPO=${GLM_MODEL_REPO:-brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78}
GLM_MODEL_REVISION=${GLM_MODEL_REVISION:-7c73450f05a151439d0f184f216b1eefcc394a31}
LOG_ROOT=${LOG_ROOT:-${RUN_ROOT}/logs}
POLL_SECONDS=${POLL_SECONDS:-30}
ACTION=${ACTION:-plan}
PUBLISH=${PUBLISH:-0}

case "${ACTION}" in
    plan|execute) ;;
    *) printf 'ACTION must be plan or execute\n' >&2; exit 2 ;;
esac
if [[ "${PUBLISH}" != 0 && "${PUBLISH}" != 1 ]]; then
    printf 'PUBLISH must be 0 or 1\n' >&2
    exit 2
fi
if [[ "${ACTION}" == plan ]]; then
    printf '{"action":"plan","code_root":"%s","publish":%s,"run_root":"%s","validation_root":"%s"}\n' \
        "${CODE_ROOT}" "${PUBLISH}" "${RUN_ROOT}" "${VALIDATION_ROOT}"
    exit 0
fi

export PYTHONPATH="${CODE_ROOT}/src:${CODE_ROOT}/scripts:${BASE_ROOT}/encoding-site${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${LOG_ROOT}" "${VALIDATION_ROOT}"

exec 9>"${LOG_ROOT}/progressive-validation.lock"
if ! flock -n 9; then
    printf 'another progressive candidate validation driver holds the lock\n' >&2
    exit 1
fi

count_encode_receipts() {
    find "${RUN_ROOT}/fast-encode" -name encode-receipt.json -type f 2>/dev/null | wc -l
}

while true; do
    receipt_count=$(count_encode_receipts)
    driver_alive=false
    if test -s "${DRIVER_PID_FILE}" && kill -0 "$(cat "${DRIVER_PID_FILE}")" 2>/dev/null; then
        driver_alive=true
    fi
    printf '{"stage":"wait-encode","receipts":%s,"driver_alive":%s}\n' \
        "${receipt_count}" "${driver_alive}"
    if test "${receipt_count}" -eq 48 && test "${driver_alive}" = false; then
        break
    fi
    if test "${driver_alive}" = false && test "${receipt_count}" -lt 48; then
        printf 'progressive driver exited with only %s/48 encode receipts\n' "${receipt_count}" >&2
        exit 1
    fi
    sleep "${POLL_SECONDS}"
done

test -s "${RUN_ROOT}/fast-k34-kld-sdpa/kld-report.json"
for layer in $(seq 0 47); do
    test -s "${RUN_ROOT}/fast-encode/layer-$(printf '%03d' "${layer}")/encode-receipt.json"
    test "$(cat "${LOG_ROOT}/fast-encode-layer-$(printf '%03d' "${layer}").exit")" = 0
done

inventory="${VALIDATION_ROOT}/candidate-inventory-local.json"
if ! test -s "${inventory}"; then
    "${PYTHON}" "${CODE_ROOT}/scripts/build_qwen_candidate_inventory.py" \
        --local-root "${RUN_ROOT}/fast-encode" \
        --output "${inventory}" \
        --workers 12 \
        --execute \
        2>&1 | tee "${LOG_ROOT}/progressive-local-candidate-inventory.log"
fi

allocation="${VALIDATION_ROOT}/frozen-causal-rate-allocation.json"
if ! test -s "${allocation}"; then
    "${PYTHON}" "${CODE_ROOT}/scripts/rebase_qwen_allocation_candidate_factory.py" \
        --source-allocation "${SOURCE_ALLOCATION}" \
        --candidate-inventory "${inventory}" \
        --encode-root "${RUN_ROOT}/fast-encode" \
        --output "${allocation}" \
        --execute \
        2>&1 | tee "${LOG_ROOT}/progressive-frozen-rate-rebase.log"
fi

union_output="${VALIDATION_ROOT}/factory-union-native-vs-progressive"
if ! test -s "${union_output}/report.json"; then
    "${PYTHON}" "${CODE_ROOT}/scripts/measure_qwen_mcg_factory_union.py" \
        --source-model "${MODEL}" \
        --model-revision "${MODEL_REVISION}" \
        --baseline-allocation "${SOURCE_ALLOCATION}" \
        --baseline-inventory "${BASELINE_INVENTORY}" \
        --baseline-label native-source-state-mcg \
        --challenger-allocation "${allocation}" \
        --challenger-inventory "${inventory}" \
        --challenger-local-root "${RUN_ROOT}/fast-encode" \
        --challenger-label progressive-state-mcg \
        --candidate-cache "${RUN_ROOT}/baseline-candidate-cache" \
        --panel-root "${TEACHER_PANEL_ROOT}" \
        --teacher-receipt "${TEACHER_PANEL_ROOT}/teacher-receipt.json" \
        --output "${union_output}" \
        --attention-backend sdpa \
        --selection-row 0 \
        --seed 20260825 \
        --execute \
        2>&1 | tee "${LOG_ROOT}/progressive-native-factory-union.log"
fi

union_verification="${union_output}/independent-verification.json"
if ! test -s "${union_verification}"; then
    # Saved-logit replay only: this starts no model and performs no additional
    # forward pass.  It binds the chosen factory allocation, teacher receipt,
    # every endpoint logit file, and every producer token-KLD array.
    "${PYTHON}" "${CODE_ROOT}/scripts/verify_qwen_candidate_factory_union.py" \
        --experiment-root "${union_output}" \
        --panel-root "${TEACHER_PANEL_ROOT}" \
        --output "${union_verification}" \
        2>&1 | tee "${LOG_ROOT}/progressive-native-factory-union-verification.log"
fi

lineage_output="${VALIDATION_ROOT}/glm-lineage"
if ! test -s "${lineage_output}/lineage.json"; then
    "${PYTHON}" "${CODE_ROOT}/scripts/bind_qwen_glm_lineage.py" \
        --control-bundle "${CONTROL_BUNDLE}" \
        --fast-kld-root "${RUN_ROOT}/fast-k34-kld-sdpa" \
        --qwen-dataset-repo "${HF_REPO}" \
        --qwen-dataset-revision "${BASE_CONTROL_REVISION}" \
        --qwen-control-prefix "${BASE_CONTROL_PREFIX}" \
        --glm-model-repo "${GLM_MODEL_REPO}" \
        --glm-model-revision "${GLM_MODEL_REVISION}" \
        --glm-calibration-path calibration/reap_recall_calib.jsonl \
        --output "${lineage_output}" \
        --execute \
        2>&1 | tee "${LOG_ROOT}/progressive-glm-lineage.log"
fi

summary_output="${VALIDATION_ROOT}/result-summary.json"
if ! test -s "${summary_output}"; then
    "${PYTHON}" "${CODE_ROOT}/scripts/summarize_qwen_progressive_candidate_result.py" \
        --native-panel-report /artifacts/shapleymcg/qwen3-30b-a3b-v1/causal-arm-v3/panel-sdpa-causal/kld-report.json \
        --fast-progressive-report "${RUN_ROOT}/fast-k34-kld-sdpa/kld-report.json" \
        --factory-union-report "${union_output}/report.json" \
        --lineage "${lineage_output}/lineage.json" \
        --output "${summary_output}" \
        --execute \
        2>&1 | tee "${LOG_ROOT}/progressive-result-summary.log"
fi

if [[ "${PUBLISH}" == 1 ]]; then
    # Publication is deliberately downstream of the completed local result.
    # A plan/execute run with the default PUBLISH=0 never reads credentials,
    # creates a remote commit, or deletes a local fit/candidate artifact.
    while pgrep -f "upload_qwen_bulk_remaining_hf.py.*--kind fit" >/dev/null; do
        printf '{"stage":"wait-fit-publication"}\n'
        sleep "${POLL_SECONDS}"
    done
    test -s "${HF_TOKEN_SOURCE}"
    token_file="${LOG_ROOT}/progressive-candidate-hf-token"
    install -m 600 "${HF_TOKEN_SOURCE}" "${token_file}"
    upload_log="${LOG_ROOT}/hf-upload-progressive-candidates.log"
    "${PYTHON}" "${CODE_ROOT}/scripts/upload_qwen_bulk_remaining_hf.py" \
        --repo-id "${HF_REPO}" \
        --run-root "${RUN_ROOT}" \
        --token-file "${token_file}" \
        --kind candidate \
        --path-prefix "${HF_PATH_PREFIX}" \
        --first-layer 0 \
        --layers 48 \
        --batch-layers 4 \
        --retry-minutes 75 \
        --include-receipted \
        2>&1 | tee "${upload_log}"

    candidate_revision=$("${PYTHON}" - "${upload_log}" <<'PY'
import json
import pathlib
import sys

for line in reversed(pathlib.Path(sys.argv[1]).read_text(errors="replace").splitlines()):
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if row.get("ok") is True and row.get("kind") == "candidate":
        print(row["receipt_revision"])
        break
else:
    raise SystemExit("candidate upload log lacks a successful immutable revision")
PY
    )
    published_inventory="${VALIDATION_ROOT}/candidate-inventory-published.json"
    if ! test -s "${published_inventory}"; then
        "${PYTHON}" "${CODE_ROOT}/scripts/build_qwen_candidate_inventory.py" \
            --repo "${HF_REPO}" \
            --revision "${candidate_revision}" \
            --path-prefix "${HF_PATH_PREFIX}" \
            --output "${published_inventory}" \
            --workers 12 \
            --execute \
            2>&1 | tee "${LOG_ROOT}/progressive-published-candidate-inventory.log"
    fi
fi

"${PYTHON}" "${CODE_ROOT}/scripts/seal_artifact_tree.py" \
    --root "${VALIDATION_ROOT}" \
    --label qwen3-30b-a3b-base-final-native-progressive-factory-union-sdpa-20k \
    --execute \
    2>&1 | tee "${LOG_ROOT}/progressive-frozen-rate-seal.log"

if [[ "${PUBLISH}" == 0 ]]; then
    printf 'final progressive factory-union validation completed locally; publication disabled\n'
    exit 0
fi

publication_root="${RUN_ROOT}/artifacts/hf-upload/progressive-validation"
mkdir -p "${publication_root}"
publication_receipt="${publication_root}/publication-receipt.json"
if ! test -s "${publication_receipt}"; then
    "${PYTHON}" "${CODE_ROOT}/scripts/upload_sealed_artifact_tree_hf.py" \
        --repo "${HF_REPO}" \
        --repo-type dataset \
        --local-root "${VALIDATION_ROOT}" \
        --path-in-repo "${HF_RESULT_PREFIX}" \
        --receipt "${publication_receipt}" \
        --commit-message "Publish Qwen progressive candidate validation" \
        --execute \
        2>&1 | tee "${LOG_ROOT}/progressive-validation-hf-upload.log"
fi

data_revision=$("${PYTHON}" - "${publication_receipt}" <<'PY'
import json
import pathlib
import sys

print(json.loads(pathlib.Path(sys.argv[1]).read_text())["revision"])
PY
)
verification_receipt="${publication_root}/remote-verification.json"
if ! test -s "${verification_receipt}"; then
    "${PYTHON}" "${CODE_ROOT}/scripts/verify_hf_artifact_tree.py" \
        --repo "${HF_REPO}" \
        --repo-type dataset \
        --revision "${data_revision}" \
        --path-in-repo "${HF_RESULT_PREFIX}" \
        --local-root "${VALIDATION_ROOT}" \
        --output "${verification_receipt}" \
        --publish-output-path "${HF_RESULT_PREFIX}/REMOTE_VERIFICATION.json" \
        --publish-message "Verify Qwen progressive candidate validation" \
        --execute \
        2>&1 | tee "${LOG_ROOT}/progressive-validation-hf-verify.log"
fi

printf 'final progressive factory-union validation completed\n'
