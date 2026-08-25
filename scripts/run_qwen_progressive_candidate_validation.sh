#!/usr/bin/env bash
set -euo pipefail

# Publish and validate one complete progressive-state Qwen candidate factory.
# The source causal allocation supplies only the K3/K4 rate choices. Candidate
# payload bytes, reconstruction hashes, and student logits come from the
# progressive factory and are independently bound through an immutable Hub
# inventory before the untouched 10x2048 SDPA panel is evaluated.

RUN_ROOT=${RUN_ROOT:-/artifacts/shapleymcg/qwen3-30b-a3b-v1/progressive-candidate-v1}
BASE_ROOT=${BASE_ROOT:-/qwen-shapleymcg-run}
CODE_ROOT=${CODE_ROOT:-${BASE_ROOT}/code-next}
PYTHON=${PYTHON:-/workspace/quant-venv/bin/python}
MODEL=${MODEL:-/models/Qwen3-30B-A3B-Base}
MODEL_REVISION=${MODEL_REVISION:-1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9}
SOURCE_ALLOCATION=${SOURCE_ALLOCATION:-/artifacts/shapleymcg/qwen3-30b-a3b-v1/causal-arm-v3/allocation-stage/allocation.json}
TEACHER_PANEL_ROOT=${TEACHER_PANEL_ROOT:-/artifacts/shapleymcg/qwen3-30b-a3b-v1/causal-arm-v3/turboderp-wiki2-sdpa-teacher}
HF_REPO=${HF_REPO:-brandonmusic/shapleymcg-qwen3-30b-a3b-reproducibility}
HF_PATH_PREFIX=${HF_PATH_PREFIX:-candidate-factories/progressive-state-v1}
HF_TOKEN_SOURCE=${HF_TOKEN_SOURCE:-/root/.cache/huggingface/token}
DRIVER_PID_FILE=${DRIVER_PID_FILE:-${RUN_ROOT}/logs/progressive-pipeline.pid}
VALIDATION_ROOT=${VALIDATION_ROOT:-${RUN_ROOT}/frozen-causal-rate-validation-v1}
LOG_ROOT=${LOG_ROOT:-${RUN_ROOT}/logs}
POLL_SECONDS=${POLL_SECONDS:-30}

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

# Avoid competing Hub commits with the fit publisher. Its data are already
# preserved remotely, but its receipt commit must finish before candidate
# publication begins.
while pgrep -f "upload_qwen_bulk_remaining_hf.py.*--kind fit" >/dev/null; do
    printf '{"stage":"wait-fit-publication"}\n'
    sleep "${POLL_SECONDS}"
done

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

inventory="${VALIDATION_ROOT}/candidate-inventory.json"
if ! test -s "${inventory}"; then
    "${PYTHON}" "${CODE_ROOT}/scripts/build_qwen_candidate_inventory.py" \
        --repo "${HF_REPO}" \
        --revision "${candidate_revision}" \
        --path-prefix "${HF_PATH_PREFIX}" \
        --output "${inventory}" \
        --workers 12 \
        --execute \
        2>&1 | tee "${LOG_ROOT}/progressive-candidate-inventory.log"
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

panel_output="${VALIDATION_ROOT}/panel-sdpa-20k"
if ! test -s "${panel_output}/kld-report.json"; then
    "${PYTHON}" "${CODE_ROOT}/scripts/measure_qwen_mcg_panel_allocation.py" \
        --source-model "${MODEL}" \
        --model-revision "${MODEL_REVISION}" \
        --allocation "${allocation}" \
        --candidate-inventory "${inventory}" \
        --local-encode-root "${RUN_ROOT}/fast-encode" \
        --candidate-cache "${RUN_ROOT}/candidate-cache" \
        --panel-root "${TEACHER_PANEL_ROOT}" \
        --teacher-root "${TEACHER_PANEL_ROOT}/teacher-logits" \
        --teacher-receipt "${TEACHER_PANEL_ROOT}/teacher-receipt.json" \
        --token-key input_ids \
        --output "${panel_output}" \
        --device-map balanced \
        --attention-backend sdpa \
        --execute \
        2>&1 | tee "${LOG_ROOT}/progressive-frozen-rate-panel.log"
fi

"${PYTHON}" "${CODE_ROOT}/scripts/seal_artifact_tree.py" \
    --root "${VALIDATION_ROOT}" \
    --label qwen3-30b-a3b-base-progressive-factory-frozen-causal-rate-sdpa-20k \
    --execute \
    2>&1 | tee "${LOG_ROOT}/progressive-frozen-rate-seal.log"

printf 'progressive frozen-causal-rate validation completed\n'
