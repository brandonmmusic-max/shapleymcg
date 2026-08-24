#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=${RUN_ROOT:-/qwen-shapleymcg-run}
PYTHON=${PYTHON:-/workspace/quant-venv/bin/python}
CODE_ROOT=${CODE_ROOT:-${RUN_ROOT}/code-next}
CAPTURE_ROOT=${CAPTURE_ROOT:-${RUN_ROOT}/calibration-capture}
SOURCE_RECEIPT=${SOURCE_RECEIPT:-${RUN_ROOT}/artifacts/qwen-source-receipt.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-${RUN_ROOT}/streaming-fit}
LOG_ROOT=${LOG_ROOT:-${RUN_ROOT}/logs}
LAYERS=${LAYERS:-48}
WAVE_SIZE=${WAVE_SIZE:-8}
MODEL_REVISION=${MODEL_REVISION:-1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9}

export PYTHONPATH="${CODE_ROOT}/src:${RUN_ROOT}/encoding-site${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

complete() {
    local layer=$1
    local label
    label=$(printf '%03d' "${layer}")
    test -f "${LOG_ROOT}/streaming-fit-layer-${label}.exit" \
        && test "$(tr -d '[:space:]' < "${LOG_ROOT}/streaming-fit-layer-${label}.exit")" = 0 \
        && test -f "${OUTPUT_ROOT}/layer-${label}/fit-receipt.json"
}

launch() {
    local layer=$1
    local label device log exit_file
    label=$(printf '%03d' "${layer}")
    device=$((layer % 2))
    log="${LOG_ROOT}/streaming-fit-layer-${label}.log"
    exit_file="${LOG_ROOT}/streaming-fit-layer-${label}.exit"
    if test -f "${exit_file}"; then
        mv "${exit_file}" "${exit_file}.previous.$(date +%s)"
    fi
    (
        start=$(date +%s)
        set +e
        "${PYTHON}" "${CODE_ROOT}/scripts/run_qwen_streaming_fit.py" \
            --capture-root "${CAPTURE_ROOT}" \
            --output-dir "${OUTPUT_ROOT}/layer-${label}" \
            --source-receipt "${SOURCE_RECEIPT}" \
            --layer "${layer}" \
            --model-revision "${MODEL_REVISION}" \
            --device "cuda:${device}" \
            --execute > "${log}" 2>&1
        code=$?
        end=$(date +%s)
        printf 'elapsed_seconds=%s\n' "$((end - start))" >> "${log}"
        printf '%s\n' "${code}" > "${exit_file}"
        exit "${code}"
    ) &
    LAUNCHED_PID=$!
}

for ((wave_start = 0; wave_start < LAYERS; wave_start += WAVE_SIZE)); do
    pids=()
    wave_end=$((wave_start + WAVE_SIZE))
    if ((wave_end > LAYERS)); then
        wave_end=${LAYERS}
    fi
    for ((layer = wave_start; layer < wave_end; layer++)); do
        if complete "${layer}"; then
            continue
        fi
        label=$(printf '%03d' "${layer}")
        if pgrep -f "run_qwen_streaming_fit.py .*--layer ${layer}( |$)" >/dev/null; then
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
            if test -f "${LOG_ROOT}/streaming-fit-layer-${label}.exit"; then
                printf 'layer %s failed; see %s\n' "${label}" \
                    "${LOG_ROOT}/streaming-fit-layer-${label}.log" >&2
                exit 1
            fi
            sleep 2
        done
    done
done

printf 'all %s layers fitted successfully\n' "${LAYERS}"
