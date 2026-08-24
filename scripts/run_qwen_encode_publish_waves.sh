#!/usr/bin/env bash
set -euo pipefail

# Encode bounded layer waves, publish the corresponding fitted Hessian/vector
# artifacts, verify them on the Hub, and only then release their local copies.
# This keeps a complete 48-layer B200 run inside a 750 GB ephemeral volume.

RUN_ROOT=${RUN_ROOT:-/qwen-shapleymcg-post-run}
CODE_ROOT=${CODE_ROOT:-${RUN_ROOT}/code}
PYTHON=${PYTHON:-/workspace/quant-venv/bin/python}
MODEL=${MODEL:-/models/Qwen3-30B-A3B-post-4c446470}
SOURCE_RECEIPT=${SOURCE_RECEIPT:-${RUN_ROOT}/artifacts/source/qwen-post-source-receipt.json}
SOURCE_ROOT=${SOURCE_ROOT:-/qwen-shapleymcg-run/sources/corrected-r10-source/reproducibility/r10}
NUMERIC_CORE=${NUMERIC_CORE:-${SOURCE_ROOT}/lineage/encode_tr3_v31.py}
EXTENSION=${EXTENSION:-/qwen-shapleymcg-run/encoding-site/exllamav3_ext.cpython-311-x86_64-linux-gnu.so}
HF_REPO_ID=${HF_REPO_ID:?HF_REPO_ID is required}
HF_TOKEN_SOURCE=${HF_TOKEN_SOURCE:-/root/.cache/huggingface/token}
WAVE_SIZE=${WAVE_SIZE:-8}
LAYERS=${LAYERS:-48}
LOG_ROOT=${LOG_ROOT:-${RUN_ROOT}/logs}

if ((WAVE_SIZE < 1 || LAYERS < 1 || LAYERS > 48)); then
    printf 'WAVE_SIZE and LAYERS must describe a positive interval within 48 layers\n' >&2
    exit 2
fi
test -s "${HF_TOKEN_SOURCE}"
mkdir -p "${LOG_ROOT}"

for ((start = 0; start < LAYERS; start += WAVE_SIZE)); do
    end=$((start + WAVE_SIZE))
    ((end > LAYERS)) && end=${LAYERS}
    count=$((end - start))
    label=$(printf '%03d-%03d' "${start}" "$((end - 1))")

    RUN_ROOT="${RUN_ROOT}" \
    CODE_ROOT="${CODE_ROOT}" \
    MODEL="${MODEL}" \
    SOURCE_RECEIPT="${SOURCE_RECEIPT}" \
    SOURCE_ROOT="${SOURCE_ROOT}" \
    NUMERIC_CORE="${NUMERIC_CORE}" \
    EXTENSION="${EXTENSION}" \
    OUTPUT_ROOT="${RUN_ROOT}/fast-encode" \
    LOG_ROOT="${LOG_ROOT}" \
    START_LAYER="${start}" \
    LAYERS="${end}" \
    WAVE_SIZE="${count}" \
    PYTHON="${PYTHON}" \
        bash "${CODE_ROOT}/scripts/run_qwen_encode_waves.sh" \
        > "${LOG_ROOT}/encode-wave-${label}.log" 2>&1

    token_file="${RUN_ROOT}/.hf-fit-wave-${label}.token"
    install -m 600 "${HF_TOKEN_SOURCE}" "${token_file}"
    HF_HUB_DISABLE_XET=1 PYTHONPATH="${CODE_ROOT}/src:${PYTHONPATH:-}" \
        "${PYTHON}" "${CODE_ROOT}/scripts/upload_qwen_bulk_remaining_hf.py" \
        --repo-id "${HF_REPO_ID}" \
        --run-root "${RUN_ROOT}" \
        --token-file "${token_file}" \
        --kind fit \
        --first-layer "${start}" \
        --layers "${count}" \
        --delete-verified \
        --batch-layers 4 \
        --retry-minutes 75 \
        > "${LOG_ROOT}/hf-fit-wave-${label}.log" 2>&1

    for ((layer = start; layer < end; layer++)); do
        item=$(printf 'layer-%03d' "${layer}")
        test -s "${RUN_ROOT}/artifacts/hf-upload/fits/${item}.json"
        test ! -e "${RUN_ROOT}/streaming-fit/${item}"
    done
    printf 'encoded layers %s and released their verified fit artifacts\n' "${label}"
done

printf 'all %s layers encoded; all fit artifacts remotely verified\n' "${LAYERS}"
