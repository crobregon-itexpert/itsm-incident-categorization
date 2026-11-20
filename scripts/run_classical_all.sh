#!/usr/bin/env bash
# Full classical suite. Usage: bash scripts/run_classical_all.sh [config]   (default configs/phase1.yaml)
# Safe to re-run: finished runs are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
CFG="${1:-configs/phase1.yaml}"
# 1) Main table: selected k, title only, all models, 5 seeds
python3 -m itsm.train_classical --config "$CFG"
# 2) k sensitivity
python3 -m itsm.train_classical --config "$CFG" --k 6 8 9 12 15 --models logreg linear_svm
# 3) Upper bound: title + solution (solution is not available at ticket creation)
python3 -m itsm.train_classical --config "$CFG" --inputs title solution --models logreg linear_svm
# 4) Human routing label (Grupo Asignado) as an external target
python3 -m itsm.train_classical --config "$CFG" --target grupo --models logreg linear_svm complement_nb
