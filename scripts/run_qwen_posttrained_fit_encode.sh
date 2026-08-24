#!/usr/bin/env bash
set -euo pipefail

# Resume the post-trained experiment at the fit boundary, record fresh
# aggregate gates, then encode/publish bounded waves only after all fits pass.

RUN_ROOT=${RUN_ROOT:-/qwen-shapleymcg-post-run}
CODE_ROOT=${CODE_ROOT:-${RUN_ROOT}/code}
CAPTURE_ROOT=${CAPTURE_ROOT:-${RUN_ROOT}/calibration-capture}
SOURCE_RECEIPT=${SOURCE_RECEIPT:-${RUN_ROOT}/artifacts/source/qwen-post-source-receipt.json}
SOURCE_MODEL=${SOURCE_MODEL:-/models/Qwen3-30B-A3B-post-4c446470}
SOURCE_REVISION=${SOURCE_REVISION:-4c446470ba0aec43e22ac1128f9ffd915f338ba3}
SOURCE_ROOT=${SOURCE_ROOT:-/qwen-shapleymcg-run/sources/corrected-r10-source/reproducibility/r10}
NUMERIC_CORE=${NUMERIC_CORE:-${SOURCE_ROOT}/lineage/encode_tr3_v31.py}
EXTENSION=${EXTENSION:-/qwen-shapleymcg-run/encoding-site/exllamav3_ext.cpython-311-x86_64-linux-gnu.so}
ENCODING_SITE=${ENCODING_SITE:-/qwen-shapleymcg-run/encoding-site}
HF_REPO_ID=${HF_REPO_ID:-brandonmusic/shapleymcg-qwen3-30b-a3b-posttrained-reproducibility}
LOG_ROOT=${LOG_ROOT:-${RUN_ROOT}/logs}
PYTHON=${PYTHON:-/workspace/quant-venv/bin/python}

mkdir -p "${LOG_ROOT}"
stamp=$(date +%s)
for name in streaming-fit-waves encode-publish-waves; do
    if test -f "${LOG_ROOT}/${name}.exit"; then
        mv "${LOG_ROOT}/${name}.exit" "${LOG_ROOT}/${name}.exit.previous.${stamp}"
    fi
done

run_stage() {
    local name=$1
    shift
    set +e
    "$@" > "${LOG_ROOT}/${name}.log" 2>&1
    local code=$?
    set -e
    printf '%s\n' "${code}" > "${LOG_ROOT}/${name}.exit"
    return "${code}"
}

export PYTHONPATH="${ENCODING_SITE}:${CODE_ROOT}/src:${PYTHONPATH:-}"
run_stage streaming-fit-waves \
    env RUN_ROOT="${RUN_ROOT}" CODE_ROOT="${CODE_ROOT}" \
    CAPTURE_ROOT="${CAPTURE_ROOT}" SOURCE_RECEIPT="${SOURCE_RECEIPT}" \
    OUTPUT_ROOT="${RUN_ROOT}/streaming-fit" LOG_ROOT="${LOG_ROOT}" \
    MODEL_REVISION="${SOURCE_REVISION}" WAVE_SIZE=8 LAYERS=48 \
    PYTHON="${PYTHON}" \
    bash "${CODE_ROOT}/scripts/run_qwen_fit_waves.sh"

run_stage encode-publish-waves \
    env RUN_ROOT="${RUN_ROOT}" CODE_ROOT="${CODE_ROOT}" \
    MODEL="${SOURCE_MODEL}" SOURCE_RECEIPT="${SOURCE_RECEIPT}" \
    SOURCE_ROOT="${SOURCE_ROOT}" NUMERIC_CORE="${NUMERIC_CORE}" \
    EXTENSION="${EXTENSION}" HF_REPO_ID="${HF_REPO_ID}" \
    WAVE_SIZE=8 LAYERS=48 PYTHON="${PYTHON}" \
    bash "${CODE_ROOT}/scripts/run_qwen_encode_publish_waves.sh"

printf 'post-trained fit, encode, and fit publication complete\n'
