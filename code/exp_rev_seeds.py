"""Revision: training-seed spread of the Proposed SSE main model
(V = 65,536, U = 4, d = 64, main budget). Seeds 1-5, evaluation over
5e5 frames at 10 and 20 dB with the evaluation seed of the letter.
Output: data/rev_seeds_sse.csv

These runs were trained on a CPU (PyTorch 2.5.1) during the revision, after the
original submission. SSE(P=4) is built before train_sse sets the seed, so the
starting weights come from the generator state at that point: for the run
labeled 1 the unseeded default generator (not recorded, and different in every
process), for runs 2-5 the state left by the training and evaluation of the previous run. The labels thus set
the training samples but not the starting weights, and the run labeled 1 is
not a re-run of sse_P4.pt (exp_main.py, set_seed(1) before SSE(P=4)).
"""
import torch
from sse_lib import DATA, SSE, train_sse, eval_ser_sse, write_csv

rows = []
for seed in (1, 2, 3, 4, 5):
    m = SSE(P=4)
    p = DATA / f"rev_sse_seed{seed}.pt"
    if p.exists():
        m.load_state_dict(torch.load(p, map_location="cpu"))
    else:
        train_sse(m, seed=seed)
        torch.save(m.state_dict(), p)
    s = eval_ser_sse(m, [10.0, 20.0], frames=500_000)
    rows.append((seed, s[0], s[1]))
    print("[seed]", rows[-1], flush=True)
write_csv(DATA / "rev_seeds_sse.csv", ["seed", "ser10", "ser20"], rows)
