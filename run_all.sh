#!/usr/bin/env bash
# Run the unit tests, the full benchmark matrix and make the plots.
#   ./run_all.sh            -> 50 rounds (default)
#   ./run_all.sh 20         -> 20 rounds (quick)
set -e
cd "$(dirname "$0")"
EPOCHS=${1:-50}
python tests/test_filter.py
python tests/test_aggregators.py
python benchmark.py --epochs "$EPOCHS" --jobs 2
python benchmark_compare.py --epochs "$EPOCHS" --jobs 2
echo "Done. See results/summary.md and results/*.png"
