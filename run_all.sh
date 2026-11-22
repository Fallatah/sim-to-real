#!/bin/bash
# Regenerates every result file, statistic and figure. Order matters (base first).
set -e
cd "$(dirname "$0")"
mkdir -p out fig
T0=$(date +%s)
[ -f out/base.pkl ] || python experiments.py base          # base ensemble + source sample
python experiments.py DR
for e in E1 E2 E2b E3 E6 E7; do python experiments.py $e; done
[ -f out/E8.csv.done ] || { python experiments.py E8 && touch out/E8.csv.done; }
[ -f out/base1.pkl ] || python experiments.py base 1        # independent bases (multi-base check)
[ -f out/base2.pkl ] || python experiments.py base 2
TEST_SEED=54321 OUT_TAG=_b0 python experiments.py E9
BASE_FILE=base1.pkl TEST_SEED=54321 OUT_TAG=_b1 python experiments.py E9
BASE_FILE=base2.pkl TEST_SEED=54321 OUT_TAG=_b2 python experiments.py E9
python extras.py
python analysis.py
echo "total wall-clock: $(( $(date +%s) - T0 )) s"
