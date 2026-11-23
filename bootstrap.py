"""How much of a difference between runs could be chance?

Resample the letters with replacement many times and recompute each
score. The middle 95% of those scores is a rough confidence interval.
No model calls: it reuses the saved runs in results/.

    python3 bootstrap.py results/dev-gemma4-e4b-v1.json results/dev-gemma4-12b-v3.json
"""
import json
import random
import sys


def urgent_caught(rows):
    urg = [r for r in rows if r["true"]["urgency"] == "urgent"]
    return sum(1 for r in urg if r["pred"] and r["pred"]["urgency"] == "urgent") / len(urg)


def both_right(rows):
    return sum(1 for r in rows if r["pred"] and r["pred"]["specialty"] == r["true"]["specialty"]
               and r["pred"]["urgency"] == r["true"]["urgency"]) / len(rows)


def ci(values):
    values = sorted(values)
    return values[int(0.025 * len(values))], values[int(0.975 * len(values))]


def main(paths, n=5000, seed=1):
    runs = {p: {r["id"]: r for r in json.load(open(p))["rows"]} for p in paths}
    ids = sorted(next(iter(runs.values())))
    rng = random.Random(seed)
    samples = {p: {"urgent": [], "both": []} for p in paths}
    diffs = {p: [] for p in paths[1:]}
    for _ in range(n):
        pick = [rng.choice(ids) for _ in ids]  # same letters for every run: a paired comparison
        base = None
        for p in paths:
            rows = [runs[p][i] for i in pick]
            if not any(r["true"]["urgency"] == "urgent" for r in rows):
                continue
            u = urgent_caught(rows)
            samples[p]["urgent"].append(u)
            samples[p]["both"].append(both_right(rows))
            if base is None:
                base = u
            else:
                diffs[p].append(u - base)
    for p in paths:
        lo, hi = ci(samples[p]["urgent"])
        blo, bhi = ci(samples[p]["both"])
        print(f"{p}: urgent caught 95% range {lo:.0%} to {hi:.0%}; both right {blo:.0%} to {bhi:.0%}")
    for p, d in diffs.items():
        lo, hi = ci(d)
        better = sum(1 for x in d if x > 0) / len(d)
        print(f"  vs first run, urgent caught difference {lo:+.0%} to {hi:+.0%}; better in {better:.0%} of resamples")


if __name__ == "__main__":
    main(sys.argv[1:])
