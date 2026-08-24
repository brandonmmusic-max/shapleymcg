#!/usr/bin/env bash
set -euo pipefail

# Evaluate the completed post-trained Qwen encode, publish every candidate,
# seal the compact result/provenance bundle, and verify that bundle on the Hub.
# This script intentionally starts only after encode-publish-waves.exit is zero.

RUN_ROOT=${RUN_ROOT:-/qwen-shapleymcg-post-run}
CODE_ROOT=${CODE_ROOT:-${RUN_ROOT}/code}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/artifacts/shapleymcg/qwen3-30b-a3b-post-v1}
PYTHON=${PYTHON:-/workspace/quant-venv/bin/python}
SOURCE_MODEL=${SOURCE_MODEL:-/models/Qwen3-30B-A3B-post-4c446470}
SOURCE_REVISION=${SOURCE_REVISION:-4c446470ba0aec43e22ac1128f9ffd915f338ba3}
TURBODERP_MODEL=${TURBODERP_MODEL:-/models/turboderp-Qwen3-30B-A3B-exl3-K4}
EXLLAMAV3_ROOT=${EXLLAMAV3_ROOT:-/qwen-shapleymcg-run/sources/exllamav3-v0.0.43}
ENCODING_SITE=${ENCODING_SITE:-/qwen-shapleymcg-run/encoding-site}
HF_REPO_ID=${HF_REPO_ID:-brandonmusic/shapleymcg-qwen3-30b-a3b-posttrained-reproducibility}
HF_TOKEN_SOURCE=${HF_TOKEN_SOURCE:-/root/.cache/huggingface/token}
LOG_ROOT=${LOG_ROOT:-${RUN_ROOT}/logs}
GPU=${GPU:-0}

test "$(tr -d '[:space:]' < "${LOG_ROOT}/encode-publish-waves.exit")" = 0
test -s "${HF_TOKEN_SOURCE}"
mkdir -p "${LOG_ROOT}"

run_stage() {
    local name=$1
    shift
    local exit_file="${LOG_ROOT}/${name}.exit"
    rm -f "${exit_file}"
    set +e
    "$@" > "${LOG_ROOT}/${name}.log" 2>&1
    local code=$?
    set -e
    printf '%s\n' "${code}" > "${exit_file}"
    return "${code}"
}

# Once every fit has been verified on the Hub and every candidate has been
# emitted, these raw activation chunks are reproducible scratch. Preserve the
# sealed manifests and receipts while releasing enough space for nine complete
# 10x2048 student-logit arms.
for ((layer = 0; layer < 48; layer++)); do
    item=$(printf 'layer-%03d' "${layer}")
    test -s "${RUN_ROOT}/artifacts/hf-upload/fits/${item}.json"
    test -s "${RUN_ROOT}/fast-encode/${item}/encode-receipt.json"
done
find "${RUN_ROOT}/calibration-capture/fit" \
    -mindepth 1 -maxdepth 1 -type d -name 'layer-[0-9][0-9][0-9]' \
    -exec rm -rf -- {} +

export PYTHONPATH="${ENCODING_SITE}:${CODE_ROOT}/src:${PYTHONPATH:-}"
run_stage matched-k4-evaluation \
    env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    "${CODE_ROOT}/scripts/measure_qwen_turboderp_hybrid_k4.py" \
    --source-model "${SOURCE_MODEL}" \
    --source-revision "${SOURCE_REVISION}" \
    --source-receipt "${RUN_ROOT}/artifacts/source/qwen-post-source-receipt.json" \
    --encode-root "${RUN_ROOT}/fast-encode" \
    --panel-root "${ARTIFACT_ROOT}/turboderp-wiki2-teacher" \
    --turboderp-model "${TURBODERP_MODEL}" \
    --turboderp-receipt "${RUN_ROOT}/artifacts/source/turboderp-k4-source-receipt.json" \
    --lineage-receipt "${RUN_ROOT}/artifacts/source/turboderp-source-lineage.json" \
    --exllamav3-root "${EXLLAMAV3_ROOT}" \
    --output "${ARTIFACT_ROOT}/matched-k4-comparison" \
    --workers 8 \
    --attention-backend eager \
    --execute

run_stage naive-controls \
    env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    "${CODE_ROOT}/scripts/measure_qwen_naive_mixed_controls.py" \
    --source-model "${SOURCE_MODEL}" \
    --encode-root "${RUN_ROOT}/fast-encode" \
    --panel-root "${ARTIFACT_ROOT}/turboderp-wiki2-teacher" \
    --selected-report "${ARTIFACT_ROOT}/matched-k4-comparison/ours-selected-k34/kld-report.json" \
    --output "${ARTIFACT_ROOT}/naive-3p5-controls-v1" \
    --seeds 0 1 2 3 4 \
    --workers 8 \
    --attention-backend eager \
    --execute

candidate_token="${RUN_ROOT}/.hf-candidates.token"
install -m 600 "${HF_TOKEN_SOURCE}" "${candidate_token}"
run_stage hf-candidates \
    env HF_HUB_DISABLE_XET=1 "${PYTHON}" \
    "${CODE_ROOT}/scripts/upload_qwen_bulk_remaining_hf.py" \
    --repo-id "${HF_REPO_ID}" \
    --run-root "${RUN_ROOT}" \
    --token-file "${candidate_token}" \
    --kind candidate \
    --kld-exit "${LOG_ROOT}/matched-k4-evaluation.exit" \
    --delete-verified \
    --batch-layers 4 \
    --retry-minutes 75

git_revision=$(git -C "${CODE_ROOT}" rev-parse HEAD)
run_stage seal-posttrained-results \
    "${PYTHON}" "${CODE_ROOT}/scripts/seal_qwen_posttrained_result_bundle.py" \
    --run-root "${RUN_ROOT}" \
    --artifact-root "${ARTIFACT_ROOT}" \
    --code-root "${CODE_ROOT}" \
    --output "${ARTIFACT_ROOT}/publication-bundle" \
    --git-revision "${git_revision}" \
    --execute

result_token="${RUN_ROOT}/.hf-results.token"
install -m 600 "${HF_TOKEN_SOURCE}" "${result_token}"
run_stage hf-posttrained-results \
    env HF_HUB_DISABLE_XET=1 "${PYTHON}" \
    "${CODE_ROOT}/scripts/upload_qwen_control_bundle_hf.py" \
    --repo-id "${HF_REPO_ID}" \
    --bundle "${ARTIFACT_ROOT}/publication-bundle" \
    --token-file "${result_token}" \
    --path-in-repo results/qwen3-30b-a3b-posttrained-v1 \
    --receipt "${ARTIFACT_ROOT}/posttrained-results-publication-receipt.json" \
    --receipt-path-in-repo receipts/qwen3-30b-a3b-posttrained-results-v1.json \
    --receipt-schema quant-pipeline.qwen-posttrained-results-hf-publication.v1 \
    --label 'Qwen3-30B-A3B post-trained matched K4 results'

printf 'post-trained evaluation and verified Hugging Face publication complete\n'
