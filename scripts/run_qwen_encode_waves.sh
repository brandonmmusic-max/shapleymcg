#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=${RUN_ROOT:-/qwen-shapleymcg-run}
PYTHON=${PYTHON:-/workspace/quant-venv/bin/python}
CODE_ROOT=${CODE_ROOT:-${RUN_ROOT}/code-next}
OUTPUT_ROOT=${OUTPUT_ROOT:-${RUN_ROOT}/fast-encode}
LOG_ROOT=${LOG_ROOT:-${RUN_ROOT}/logs}
LAYERS=${LAYERS:-48}
START_LAYER=${START_LAYER:-0}
WAVE_SIZE=${WAVE_SIZE:-4}
MODEL=${MODEL:-/models/Qwen3-30B-A3B-Base}
SOURCE_ROOT=${SOURCE_ROOT:-${RUN_ROOT}/sources/corrected-r10-source/reproducibility/r10}
NUMERIC_CORE=${NUMERIC_CORE:-${SOURCE_ROOT}/lineage/encode_tr3_v31.py}
EXTENSION=${EXTENSION:-${RUN_ROOT}/encoding-site/exllamav3_ext.cpython-311-x86_64-linux-gnu.so}

export PYTHONPATH="${CODE_ROOT}/src:${RUN_ROOT}/encoding-site${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

complete() {
    local layer=$1 label
    label=$(printf '%03d' "${layer}")
    test -f "${LOG_ROOT}/fast-encode-layer-${label}.exit" \
        && test "$(tr -d '[:space:]' < "${LOG_ROOT}/fast-encode-layer-${label}.exit")" = 0 \
        && test -f "${OUTPUT_ROOT}/layer-${label}/encode-receipt.json"
}

launch() {
    local layer=$1 label gpu log exit_file
    label=$(printf '%03d' "${layer}")
    gpu=$((layer % 2))
    log="${LOG_ROOT}/fast-encode-layer-${label}.log"
    exit_file="${LOG_ROOT}/fast-encode-layer-${label}.exit"
    if test -f "${exit_file}"; then
        mv "${exit_file}" "${exit_file}.previous.$(date +%s)"
    fi
    (
        export CUDA_VISIBLE_DEVICES=${gpu}
        set +e
        "${PYTHON}" "${CODE_ROOT}/scripts/run_qwen_fast_encode.py" \
            --model "${MODEL}" \
            --fit-root "${RUN_ROOT}/streaming-fit" \
            --layer "${layer}" \
            --output "${OUTPUT_ROOT}/layer-${label}" \
            --source-root "${SOURCE_ROOT}" \
            --numeric-core "${NUMERIC_CORE}" \
            --extension "${EXTENSION}" \
            --device cuda:0 \
            --execute > "${log}" 2>&1
        code=$?
        printf '%s\n' "${code}" > "${exit_file}"
        exit "${code}"
    ) &
    LAUNCHED_PID=$!
}

if ((START_LAYER < 0 || START_LAYER >= LAYERS)); then
    printf 'START_LAYER must be in [0, %s)\n' "${LAYERS}" >&2
    exit 2
fi

for ((wave_start = START_LAYER; wave_start < LAYERS; wave_start += WAVE_SIZE)); do
    pids=()
    wave_end=$((wave_start + WAVE_SIZE))
    ((wave_end > LAYERS)) && wave_end=${LAYERS}
    for ((layer = wave_start; layer < wave_end; layer++)); do
        complete "${layer}" && continue
        if pgrep -f "run_qwen_fast_encode.py .*--layer ${layer}( |$)" >/dev/null; then
            continue
        fi
        launch "${layer}"
        pids+=("${LAUNCHED_PID}")
    done
    for pid in "${pids[@]}"; do
        wait "${pid}"
    done
    for ((layer = wave_start; layer < wave_end; layer++)); do
        until complete "${layer}"; do
            label=$(printf '%03d' "${layer}")
            if test -f "${LOG_ROOT}/fast-encode-layer-${label}.exit"; then
                printf 'layer %s failed; see %s\n' "${label}" \
                    "${LOG_ROOT}/fast-encode-layer-${label}.log" >&2
                exit 1
            fi
            sleep 2
        done
    done
done

printf 'all %s layers encoded successfully\n' "${LAYERS}"
