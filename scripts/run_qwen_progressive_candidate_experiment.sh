#!/usr/bin/env bash
set -euo pipefail

# Continue the sealed progressive-state candidate experiment after its routed
# fit capture completes.  The capture model is a verified causal reconstruction;
# every encode still reads weights from the immutable BF16 source checkpoint.

RUN_ROOT=${RUN_ROOT:-/workspace/shapleymcg-progressive-v1}
BASE_ROOT=${BASE_ROOT:-/qwen-shapleymcg-run}
SCRIPT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CODE_ROOT=${CODE_ROOT:-${SCRIPT_ROOT}}
PYTHON=${PYTHON:-/workspace/quant-venv/bin/python}
MODEL=${MODEL:-/models/Qwen3-30B-A3B-Base}
SOURCE_RECEIPT=${SOURCE_RECEIPT:-${BASE_ROOT}/artifacts/qwen-source-receipt.json}
SOURCE_ROOT=${SOURCE_ROOT:-${BASE_ROOT}/sources/corrected-r10-source/reproducibility/r10}
NUMERIC_CORE=${NUMERIC_CORE:-${SOURCE_ROOT}/lineage/encode_tr3_v31.py}
EXTENSION=${EXTENSION:-${BASE_ROOT}/encoding-site/exllamav3_ext.cpython-311-x86_64-linux-gnu.so}
KLD_WINDOW=${KLD_WINDOW:-/artifacts/shapleymcg/qwen3-30b-a3b-v1/kld-window}
TEACHER=${TEACHER:-/artifacts/shapleymcg/qwen3-30b-a3b-v1/teacher-kld/window-0000.safetensors}
LOG_ROOT=${LOG_ROOT:-${RUN_ROOT}/logs}
ACTION=${ACTION:-plan}
CAPTURE_MODE=${CAPTURE_MODE:-require}
SEALED_CORPUS=${SEALED_CORPUS:-}
CAPTURE_RECEIPT=${CAPTURE_RECEIPT:-${RUN_ROOT}/calibration-capture/calibration-capture-fit-receipt.json}
CAPTURE_DEVICE_MAP=${CAPTURE_DEVICE_MAP:-auto}
CAPTURE_ATTENTION_BACKEND=${CAPTURE_ATTENTION_BACKEND:-eager}

case "${ACTION}" in
    plan|execute) ;;
    *) printf 'ACTION must be plan or execute\n' >&2; exit 2 ;;
esac
case "${CAPTURE_MODE}" in
    launch|require) ;;
    *) printf 'CAPTURE_MODE must be launch or require\n' >&2; exit 2 ;;
esac

if [[ "${ACTION}" == plan ]]; then
    printf '{"action":"plan","capture_mode":"%s","capture_receipt":"%s","code_root":"%s","publication":false,"run_root":"%s"}\n' \
        "${CAPTURE_MODE}" "${CAPTURE_RECEIPT}" "${CODE_ROOT}" "${RUN_ROOT}"
    exit 0
fi

export PYTHONPATH="${CODE_ROOT}/src:${BASE_ROOT}/encoding-site${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${LOG_ROOT}"

if ! test -s "${CAPTURE_RECEIPT}"; then
    if [[ "${CAPTURE_MODE}" == require ]]; then
        printf 'required capture receipt is absent: %s\n' "${CAPTURE_RECEIPT}" >&2
        exit 1
    fi
    if [[ -z "${SEALED_CORPUS}" ]]; then
        printf 'SEALED_CORPUS is required when CAPTURE_MODE=launch\n' >&2
        exit 2
    fi
    capture_args=(
        "${PYTHON}" "${CODE_ROOT}/scripts/run_qwen_calibration_capture.py"
        --source-checkpoint "${MODEL}"
        --sealed-corpus "${SEALED_CORPUS}"
        --output-dir "${RUN_ROOT}/calibration-capture"
        --model-revision "${MODEL_REVISION:-1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9}"
        --device-map "${CAPTURE_DEVICE_MAP}"
        --attention-backend "${CAPTURE_ATTENTION_BACKEND}"
        --purposes fit
        --execute
    )
    "${capture_args[@]}" > "${LOG_ROOT}/capture-fit.log" 2>&1
fi
test -s "${CAPTURE_RECEIPT}"

# The independent candidate-factory union uses the second B200 while capture
# runs on the first.  Do not start two-GPU fit waves until that orthogonal arm
# has released its model replica.
union_pid_file="${LOG_ROOT}/factory-union.pid"
while test -s "${union_pid_file}" && kill -0 "$(cat "${union_pid_file}")" 2>/dev/null; do
    sleep 30
done

RUN_ROOT="${RUN_ROOT}" \
CODE_ROOT="${CODE_ROOT}" \
PYTHON="${PYTHON}" \
CAPTURE_ROOT="${RUN_ROOT}/calibration-capture" \
SOURCE_RECEIPT="${SOURCE_RECEIPT}" \
OUTPUT_ROOT="${RUN_ROOT}/streaming-fit" \
LOG_ROOT="${LOG_ROOT}" \
WAVE_SIZE=8 \
    bash "${CODE_ROOT}/scripts/run_qwen_fit_waves.sh" \
    > "${LOG_ROOT}/progressive-fit-waves.log" 2>&1

RUN_ROOT="${RUN_ROOT}" \
CODE_ROOT="${CODE_ROOT}" \
PYTHON="${PYTHON}" \
MODEL="${MODEL}" \
SOURCE_RECEIPT="${SOURCE_RECEIPT}" \
SOURCE_ROOT="${SOURCE_ROOT}" \
NUMERIC_CORE="${NUMERIC_CORE}" \
EXTENSION="${EXTENSION}" \
OUTPUT_ROOT="${RUN_ROOT}/fast-encode" \
LOG_ROOT="${LOG_ROOT}" \
WAVE_SIZE=4 \
    bash "${CODE_ROOT}/scripts/run_qwen_encode_waves.sh" \
    > "${LOG_ROOT}/progressive-encode-waves.log" 2>&1

"${PYTHON}" "${CODE_ROOT}/scripts/run_qwen_fast_kld.py" \
    --model "${MODEL}" \
    --encode-root "${RUN_ROOT}/fast-encode" \
    --kld-window "${KLD_WINDOW}" \
    --teacher "${TEACHER}" \
    --output "${RUN_ROOT}/fast-k34-kld-sdpa" \
    --attention-backend sdpa \
    --execute > "${LOG_ROOT}/progressive-fast-kld.log" 2>&1

printf 'progressive candidate fit, encode, and first KLD measurement completed\n'
