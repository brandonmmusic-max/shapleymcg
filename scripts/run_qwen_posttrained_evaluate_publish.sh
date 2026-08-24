#!/usr/bin/env bash
set -euo pipefail

# Evaluate the completed post-trained Qwen encode, publish every candidate,
# seal the compact result/provenance bundle, and verify that bundle on the Hub.
# KLD may start as soon as all 48 sealed encode receipts exist.  Fit/candidate
# publication remains behind the aggregate encode-publish gate.

RUN_ROOT=${RUN_ROOT:-/qwen-shapleymcg-post-run}
CODE_ROOT=${CODE_ROOT:-${RUN_ROOT}/code}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/artifacts/shapleymcg/qwen3-30b-a3b-post-v1}
CAPTURE_ROOT=${CAPTURE_ROOT:-${RUN_ROOT}/calibration-capture-route-complete}
PYTHON=${PYTHON:-/workspace/quant-venv/bin/python}
export PATH="$(dirname "${PYTHON}"):${PATH}"
SOURCE_MODEL=${SOURCE_MODEL:-/models/Qwen3-30B-A3B-post-4c446470}
SOURCE_REVISION=${SOURCE_REVISION:-4c446470ba0aec43e22ac1128f9ffd915f338ba3}
TURBODERP_MODEL=${TURBODERP_MODEL:-/models/turboderp-Qwen3-30B-A3B-exl3-K4}
EXLLAMAV3_ROOT=${EXLLAMAV3_ROOT:-/qwen-shapleymcg-run/sources/exllamav3-v0.0.43}
ENCODING_SITE=${ENCODING_SITE:-/qwen-shapleymcg-run/encoding-site}
HF_REPO_ID=${HF_REPO_ID:-brandonmusic/shapleymcg-qwen3-30b-a3b-posttrained-reproducibility}
HF_TOKEN_SOURCE=${HF_TOKEN_SOURCE:-/root/.cache/huggingface/token}
LOG_ROOT=${LOG_ROOT:-${RUN_ROOT}/logs}
GPU=${GPU:-0}
WAIT_FOR_ENCODE=${WAIT_FOR_ENCODE:-0}
WAIT_FOR_LAYER_ENCODES=${WAIT_FOR_LAYER_ENCODES:-0}

mkdir -p "${LOG_ROOT}"
on_exit() {
    local code=$?
    trap - EXIT
    printf '%s\n' "${code}" > "${LOG_ROOT}/posttrained-evaluate-publish.exit"
    exit "${code}"
}
trap on_exit EXIT

wait_for_encode_publication() {
    while ! test -f "${LOG_ROOT}/encode-publish-waves.exit"; do
        sleep 15
    done
    test "$(tr -d '[:space:]' < "${LOG_ROOT}/encode-publish-waves.exit")" = 0
}

all_layer_encodes_complete() {
    local layer item exit_file
    for ((layer = 0; layer < 48; layer++)); do
        item=$(printf 'layer-%03d' "${layer}")
        exit_file="${LOG_ROOT}/fast-encode-layer-$(printf '%03d' "${layer}").exit"
        test -f "${exit_file}" || return 1
        test "$(tr -d '[:space:]' < "${exit_file}")" = 0 || return 2
        test -s "${RUN_ROOT}/fast-encode/${item}/encode-receipt.json" || return 1
    done
}

if [[ "${WAIT_FOR_ENCODE}" != 0 && "${WAIT_FOR_ENCODE}" != 1 ]]; then
    printf 'WAIT_FOR_ENCODE must be 0 or 1\n' >&2
    exit 2
fi
if [[ "${WAIT_FOR_LAYER_ENCODES}" != 0 && "${WAIT_FOR_LAYER_ENCODES}" != 1 ]]; then
    printf 'WAIT_FOR_LAYER_ENCODES must be 0 or 1\n' >&2
    exit 2
fi
if [[ "${WAIT_FOR_ENCODE}" == 1 && "${WAIT_FOR_LAYER_ENCODES}" == 1 ]]; then
    printf 'select only one encode wait mode\n' >&2
    exit 2
fi

if [[ "${WAIT_FOR_LAYER_ENCODES}" == 1 ]]; then
    while true; do
        set +e
        all_layer_encodes_complete
        layer_status=$?
        set -e
        case "${layer_status}" in
            0) break ;;
            2)
                printf 'a layer encode failed before KLD\n' >&2
                exit 1
                ;;
        esac
        sleep 15
    done
else
    wait_for_encode_publication
fi
test -s "${HF_TOKEN_SOURCE}"

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

export PYTHONPATH="${ENCODING_SITE}:${CODE_ROOT}/src:${PYTHONPATH:-}"
matched_complete() {
    test -f "${LOG_ROOT}/matched-k4-evaluation.exit" \
        && test "$(tr -d '[:space:]' < "${LOG_ROOT}/matched-k4-evaluation.exit")" = 0 \
        && test -s "${ARTIFACT_ROOT}/matched-k4-comparison/summary.json" \
        && test -s "${ARTIFACT_ROOT}/matched-k4-comparison/ours-selected-k34/kld-report.json" \
        && test -s "${ARTIFACT_ROOT}/matched-k4-comparison/ours-expert-k4/kld-report.json" \
        && test -s "${ARTIFACT_ROOT}/matched-k4-comparison/turboderp-full-k4/kld-report.json" \
        && test -s "${ARTIFACT_ROOT}/matched-k4-comparison/hybrid-ours-experts/kld-report.json"
}

if ! matched_complete; then
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
    --resume \
    --execute
else
    printf 'adopted successful matched K4 evaluation\n'
fi

if ! test -f "${LOG_ROOT}/naive-controls.exit" \
    || ! test "$(tr -d '[:space:]' < "${LOG_ROOT}/naive-controls.exit")" = 0 \
    || ! test -s "${ARTIFACT_ROOT}/naive-3p5-controls-v1/summary.json"; then
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
else
    printf 'adopted successful naive controls\n'
fi

# Publication/deletion remains behind the full remote fit gate even when KLD
# started early from sealed local encode receipts.
wait_for_encode_publication
for ((layer = 0; layer < 48; layer++)); do
    item=$(printf 'layer-%03d' "${layer}")
    test -s "${RUN_ROOT}/artifacts/hf-upload/fits/${item}.json"
    test -s "${RUN_ROOT}/fast-encode/${item}/encode-receipt.json"
done
find "${CAPTURE_ROOT}/fit" \
    -mindepth 1 -maxdepth 1 -type d -name 'layer-[0-9][0-9][0-9]' \
    -exec rm -rf -- {} +

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

git_revision=${GIT_REVISION:-$(git -C "${CODE_ROOT}" rev-parse HEAD)}
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
