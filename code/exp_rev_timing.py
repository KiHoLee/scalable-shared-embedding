"""Revision: where the eight-times wall-clock ratio comes from, and decoding time
on one CPU thread. Medians per step or token, one job at a time.
Output: data/rev_timing.csv (device, scheme, part, seconds)."""
import time

import torch

from sse_lib import LR, SSE, TRAIN_BATCH, TRAIN_SNR_DB, FullCodebook, write_csv

V = 65536


def med(x):
    return sorted(x)[len(x) // 2]


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def steps(m, kind, n, dev):
    m.to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    ce = torch.nn.CrossEntropyLoss()
    tc, tf, to = [], [], []
    for _ in range(n):
        nc = m.vu if kind == "sse" else m.Vc
        shape = (TRAIN_BATCH, m.users, m.P) if kind == "sse" else (TRAIN_BATCH, m.users)
        x = torch.randint(nc, shape, device=dev)
        snr = torch.empty(TRAIN_BATCH, device=dev).uniform_(*TRAIN_SNR_DB)
        sync(dev); t0 = time.perf_counter()
        m.calibrate_power(8192)
        sync(dev); t1 = time.perf_counter()
        loss = ce((m(x, snr) * m.logit_scale.exp()).reshape(-1, nc), x.reshape(-1))
        opt.zero_grad()
        loss.backward()
        sync(dev); t2 = time.perf_counter()
        opt.step()
        sync(dev); t3 = time.perf_counter()
        tc.append(t1 - t0); tf.append(t2 - t1); to.append(t3 - t2)
    return med(tc), med(tf), med(to)


def main():
    rows = []
    for dev_name, ns, nf in (("cuda:0", 300, 30), ("cpu", 100, 5)):
        dev = torch.device(dev_name)
        if dev.type == "cpu":
            torch.set_num_threads(1)
        for kind, n, m in (("sse", ns, SSE(P=4, users=4)), ("full", nf, FullCodebook(V, users=4))):
            c, f, o = steps(m, kind, n, dev)
            for part, v in (("calibrate", c), ("forward_backward", f), ("optimizer", o),
                            ("step_total", c + f + o)):
                rows.append((dev_name, kind, part, v))
                print(rows[-1], flush=True)
                write_csv(DATA_OUT, ["device", "scheme", "part", "seconds"], rows)
    torch.set_num_threads(1)
    B, mu, r = torch.randn(16, 16), torch.randn(16), torch.randn(4, 16)
    C, rr = torch.randn(V, 64) * torch.randn(64), torch.randn(64)
    cand = B * mu

    def t(fn, n):
        ts = []
        for _ in range(n):
            a = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - a)
        return med(ts)
    rows.append(("cpu", "sse", "decode_token", t(lambda: (r @ cand.T).argmax(1), 5000)))
    rows.append(("cpu", "full", "decode_token", t(lambda: (C @ rr).argmax(), 200)))
    print(rows[-2:], flush=True)
    write_csv(DATA_OUT, ["device", "scheme", "part", "seconds"], rows)


if __name__ == "__main__":
    from sse_lib import DATA
    DATA_OUT = DATA / "rev_timing.csv"
    main()
