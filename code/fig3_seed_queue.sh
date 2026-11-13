#!/bin/bash
# Seeds 4-10 of the long-budget experiment (Fig. 3, Table I): every scheme
# trained for 2e5 iterations with the recipe of seeds 2 and 3
# (exp_rev_budget.py --log-frames 200000), N_PAR runs at a time on one GPU.
# Outputs carry the tags _s4.._s10 and leave the seed 1-3 files untouched.
# The time_s columns of these runs reflect a shared GPU and are not used for
# wall-clock statements. Afterwards rebuild the ten-seed means with
#   python exp_rev_budget_eval.py --aggregate-only
# Example: PYTHON=/path/to/venv/bin/python GPU=1 N_PAR=4.
cd "$(dirname "$0")" || exit 1
PYTHON=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
LOG=${LOG:-../fig3_runs}; mkdir -p "$LOG"
N_PAR=${N_PAR:-4}
for s in 4 5 6 7 8 9 10; do
  ( "$PYTHON" exp_rev_budget.py --schemes sse,hier,sampled,full \
      --seed $s --log-frames 200000 > "$LOG/seed$s.log" 2>&1; echo "exit $?" >> "$LOG/seed$s.log" ) &
  sleep 5
  while [ "$(jobs -rp | wc -l)" -ge "$N_PAR" ]; do sleep 30; done
done
wait
date > "$LOG/ALL_DONE"
