#!/bin/bash
set -e

SUFFIX=${1}
GPU_ENV_NAME=kipy-gpu-full${SUFFIX}
mamba create -y -f gpu.yaml -f common.yaml -f kirad.yaml -n ${GPU_ENV_NAME}
mamba env export -n ${GPU_ENV_NAME} > kipy-gpu-full.yaml
CPU_ENV_NAME=kipy-cpu-full${SUFFIX}
mamba create -y -f cpu.yaml -f common.yaml -n ${CPU_ENV_NAME}
mamba env export -n ${CPU_ENV_NAME} > kipy-cpu-full.yaml
if grep "cuda" kipy-cpu-full.yaml; then
  echo "Env creation failed; cpu env pulled in cuda dependencies."
fi
