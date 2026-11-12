#!/bin/bash
# SLURM array job for the slow ng_prepare.py steps (segments / images).
#
# Submit from scripts/. Run the --init step once first (it creates the empty pyramid):
#   python ng_prepare.py segments --config ../configs/X.yaml --init
#   mkdir -p slurm_output
#   sbatch --array=0-19 ng_prepare.sbatch ../configs/X.yaml segments 20
# Each task converts T_PER_TASK timepoints: task i does [i*T_PER_TASK, (i+1)*T_PER_TASK).
# For wt (400 timepoints) segments take ~5 s per timepoint, so 20 tasks of 20 is plenty;
# a single process (no array) also finishes in about 35 minutes.

#SBATCH --job-name=ng_prepare
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_output/ng_prepare_%A_%a.out

set -euo pipefail

CONFIG=$1
STEP=$2
T_PER_TASK=${3:-20}
PYTHON=${PYTHON:-/hpc/mydata/bruno.moretti/envs/zms2-ng/bin/python}

T_START=$(( SLURM_ARRAY_TASK_ID * T_PER_TASK ))
T_STOP=$(( T_START + T_PER_TASK ))

cd "$SLURM_SUBMIT_DIR"
"$PYTHON" ng_prepare.py "$STEP" --config "$CONFIG" --t-start "$T_START" --t-stop "$T_STOP"
