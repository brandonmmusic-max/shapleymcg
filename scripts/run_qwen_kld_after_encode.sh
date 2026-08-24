#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=${RUN_ROOT:-/qwen-shapleymcg-run}
PYTHON=${PYTHON:-/workspace/quant-venv/bin/python}
CODE_ROOT=${CODE_ROOT:-${RUN_ROOT}/code-next}
ENCODE_EXIT=${ENCODE_EXIT:-${RUN_ROOT}/logs/fast-encode-waves.exit}
MODEL=${MODEL:-/models/Qwen3-30B-A3B-Base}
ENCODE_ROOT=${ENCODE_ROOT:-${RUN_ROOT}/fast-encode}
KLD_WINDOW=${KLD_WINDOW:-/artifacts/shapleymcg/qwen3-30b-a3b-v1/kld-window}
TEACHER=${TEACHER:-/artifacts/shapleymcg/qwen3-30b-a3b-v1/teacher-kld/window-0000.safetensors}
OUTPUT=${OUTPUT:-/artifacts/shapleymcg/qwen3-30b-a3b-v1/fast-k34-kld}

export PYTHONPATH="${CODE_ROOT}/src:${RUN_ROOT}/encoding-site${PYTHONPATH:+:${PYTHONPATH}}"

while ! test -f "${ENCODE_EXIT}"; do
    sleep 15
done
if test "$(tr -d '[:space:]' < "${ENCODE_EXIT}")" != 0; then
    printf 'encode scheduler failed; refusing KLD replay\n' >&2
    exit 1
fi

"${PYTHON}" "${CODE_ROOT}/scripts/run_qwen_fast_kld.py" \
    --model "${MODEL}" \
    --encode-root "${ENCODE_ROOT}" \
    --kld-window "${KLD_WINDOW}" \
    --teacher "${TEACHER}" \
    --output "${OUTPUT}" \
    --attention-backend eager \
    --execute
