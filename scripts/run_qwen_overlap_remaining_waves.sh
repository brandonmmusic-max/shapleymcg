#!/usr/bin/env bash
set -euo pipefail

# Overlap a completed wave's fit publication with the following encode wave.
# The primary encode/publish controller remains authoritative: this helper
# waits until its uploader is active (or the whole range is receipted) before
# launching disjoint layers.  run_qwen_encode_waves.sh and the primary
# controller both adopt existing live/successful layer work, so no layer is
# encoded twice.

RUN_ROOT=${RUN_ROOT:-/qwen-shapleymcg-post-run}
CODE_ROOT=${CODE_ROOT:-${RUN_ROOT}/code}
LOG_ROOT=${LOG_ROOT:-${RUN_ROOT}/logs}
PYTHON=${PYTHON:-/workspace/quant-venv/bin/python}
MODEL=${MODEL:-/models/Qwen3-30B-A3B-post-4c446470}
SOURCE_RECEIPT=${SOURCE_RECEIPT:-${RUN_ROOT}/artifacts/source/qwen-post-source-receipt.json}
SOURCE_ROOT=${SOURCE_ROOT:-/qwen-shapleymcg-run/sources/corrected-r10-source/reproducibility/r10}
NUMERIC_CORE=${NUMERIC_CORE:-${SOURCE_ROOT}/lineage/encode_tr3_v31.py}
EXTENSION=${EXTENSION:-/qwen-shapleymcg-run/encoding-site/exllamav3_ext.cpython-311-x86_64-linux-gnu.so}
CURRENT_FIRST=${CURRENT_FIRST:-24}
NEXT_FIRST=${NEXT_FIRST:-32}
FINAL_FIRST=${FINAL_FIRST:-40}
LAYERS=${LAYERS:-48}
WAVE_SIZE=${WAVE_SIZE:-8}

if ! ((0 <= CURRENT_FIRST && CURRENT_FIRST < NEXT_FIRST \
    && NEXT_FIRST < FINAL_FIRST && FINAL_FIRST < LAYERS \
    && NEXT_FIRST - CURRENT_FIRST == WAVE_SIZE \
    && FINAL_FIRST - NEXT_FIRST == WAVE_SIZE \
    && LAYERS - FINAL_FIRST == WAVE_SIZE)); then
    printf 'ranges must describe three consecutive WAVE_SIZE intervals\n' >&2
    exit 2
fi

primary_alive() {
    pgrep -f "${CODE_ROOT}/scripts/run_qwen_encode_publish_waves.sh" >/dev/null
}

range_complete() {
    local first=$1 limit=$2 layer label exit_file
    for ((layer = first; layer < limit; layer++)); do
        label=$(printf '%03d' "${layer}")
        exit_file="${LOG_ROOT}/fast-encode-layer-${label}.exit"
        test -f "${exit_file}" || return 1
        test "$(tr -d '[:space:]' < "${exit_file}")" = 0 || return 2
        test -s "${RUN_ROOT}/fast-encode/layer-${label}/encode-receipt.json" || return 1
    done
}

range_receipted() {
    local first=$1 limit=$2 layer label
    for ((layer = first; layer < limit; layer++)); do
        label=$(printf '%03d' "${layer}")
        test -s "${RUN_ROOT}/artifacts/hf-upload/fits/layer-${label}.json" || return 1
    done
}

wait_for_range() {
    local first=$1 limit=$2 code
    while true; do
        set +e
        range_complete "${first}" "${limit}"
        code=$?
        set -e
        case "${code}" in
            0) return 0 ;;
            2)
                printf 'encode failure in range %03d-%03d\n' \
                    "${first}" "$((limit - 1))" >&2
                return 2
                ;;
        esac
        primary_alive || {
            printf 'primary encode/publish controller stopped\n' >&2
            return 3
        }
        sleep 15
    done
}

wait_for_fit_publication() {
    local first=$1 limit=$2
    while true; do
        range_receipted "${first}" "${limit}" && return 0
        if pgrep -f \
            "upload_qwen_bulk_remaining_hf.py .*--kind fit .*--first-layer ${first}" \
            >/dev/null; then
            return 0
        fi
        primary_alive || {
            printf 'primary encode/publish controller stopped before fit publication\n' >&2
            return 3
        }
        sleep 5
    done
}

launch_range() {
    local first=$1 limit=$2 label
    label=$(printf '%03d-%03d' "${first}" "$((limit - 1))")
    printf 'launch overlap encode %s at %s\n' "${label}" "$(date -Is)"
    env RUN_ROOT="${RUN_ROOT}" CODE_ROOT="${CODE_ROOT}" \
        MODEL="${MODEL}" SOURCE_RECEIPT="${SOURCE_RECEIPT}" \
        SOURCE_ROOT="${SOURCE_ROOT}" NUMERIC_CORE="${NUMERIC_CORE}" \
        EXTENSION="${EXTENSION}" OUTPUT_ROOT="${RUN_ROOT}/fast-encode" \
        LOG_ROOT="${LOG_ROOT}" START_LAYER="${first}" LAYERS="${limit}" \
        WAVE_SIZE="${WAVE_SIZE}" PYTHON="${PYTHON}" \
        bash "${CODE_ROOT}/scripts/run_qwen_encode_waves.sh" \
        > "${LOG_ROOT}/overlap-encode-wave-${label}.log" 2>&1
    printf 'complete overlap encode %s at %s\n' "${label}" "$(date -Is)"
}

wait_for_range "${CURRENT_FIRST}" "${NEXT_FIRST}"
wait_for_fit_publication "${CURRENT_FIRST}" "${NEXT_FIRST}"
launch_range "${NEXT_FIRST}" "${FINAL_FIRST}"
wait_for_fit_publication "${NEXT_FIRST}" "${FINAL_FIRST}"
launch_range "${FINAL_FIRST}" "${LAYERS}"
printf 'remaining overlap launcher complete at %s\n' "$(date -Is)"
