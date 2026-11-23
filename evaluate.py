"""Run the classifier over a labelled set and score it.

    python3 evaluate.py --prompt v1 --set dev --workers 1
    python3 evaluate.py --prompt v3 --set dev --workers 1 --thinking
    LLM_MODEL=gemma4:12b python3 evaluate.py --prompt v4 --set test2 --workers 1
    python3 evaluate.py --compare results/dev-gemma4-e4b-v1.json results/dev-gemma4-12b-v4.json

Each run is saved to results/<set>-<model>-<prompt>[-thinking].json so runs
can be compared later without calling the model again. Use --workers 1 with
Ollama: it serves one request at a time, so parallel workers only queue and
inflate the per-letter timings.
"""
import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from classify import SPECIALTIES, URGENCIES, classify

HERE = Path(__file__).parent

# How bad each urgency mistake is. Under-triage (urgent sent to the back of
# the queue) is the one that hurts patients, so it costs the most.
COST = {
    ("urgent", "semi-urgent"): 3, ("urgent", "routine"): 10,
    ("semi-urgent", "routine"): 3, ("semi-urgent", "urgent"): 1,
    ("routine", "urgent"): 1, ("routine", "semi-urgent"): 1,
}


def load(name):
    return [json.loads(l) for l in (HERE / "data" / f"{name}.jsonl").open()]


def run(cases, prompt, thinking, workers):
    def one(case):
        t = time.time()
        try:
            pred = classify(case["text"], prompt=prompt, thinking=thinking)
            err = None
        except Exception as e:  # a bad reply counts as a miss, not a crash
            pred, err = None, str(e)
        return {"id": case["id"], "hard": case["hard"],
                "true": {"specialty": case["specialty"], "urgency": case["urgency"]},
                "pred": pred, "error": err, "seconds": round(time.time() - t, 2)}
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, cases))


def prf(rows, field, label):
    tp = sum(1 for r in rows if r["pred"] and r["pred"][field] == label and r["true"][field] == label)
    fp = sum(1 for r in rows if r["pred"] and r["pred"][field] == label and r["true"][field] != label)
    fn = sum(1 for r in rows if r["true"][field] == label and (not r["pred"] or r["pred"][field] != label))
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 3), "recall": round(rc, 3), "support": tp + fn}


def score(rows):
    n = len(rows)
    ok = [r for r in rows if r["pred"]]
    acc = lambda f: sum(1 for r in ok if r["pred"][f] == r["true"][f]) / n
    both = sum(1 for r in ok if r["pred"]["specialty"] == r["true"]["specialty"]
               and r["pred"]["urgency"] == r["true"]["urgency"]) / n
    confusion = Counter((r["true"]["urgency"], r["pred"]["urgency"] if r["pred"] else "error")
                        for r in rows)
    cost = sum(COST.get((r["true"]["urgency"], r["pred"]["urgency"]), 0) if r["pred"] else 10
               for r in rows)
    hard = [r for r in rows if r["hard"]]
    return {
        "n": n,
        "errors": n - len(ok),
        "specialty_accuracy": round(acc("specialty"), 3),
        "urgency_accuracy": round(acc("urgency"), 3),
        "both_correct": round(both, 3),
        "hard_both_correct": round(sum(1 for r in hard if r["pred"]
                                       and r["pred"]["specialty"] == r["true"]["specialty"]
                                       and r["pred"]["urgency"] == r["true"]["urgency"]) / max(len(hard), 1), 3),
        "urgency": {u: prf(rows, "urgency", u) for u in URGENCIES},
        "specialty": {s: prf(rows, "specialty", s) for s in SPECIALTIES},
        "urgent_missed": sum(1 for r in rows if r["true"]["urgency"] == "urgent"
                             and (not r["pred"] or r["pred"]["urgency"] != "urgent")),
        "urgent_sent_to_routine": confusion[("urgent", "routine")],
        "triage_cost": cost,
        "urgency_confusion": {f"{t} -> {p}": c for (t, p), c in sorted(confusion.items())},
        "median_seconds": sorted(r["seconds"] for r in rows)[n // 2],
    }


def review_sweep(rows):
    """If letters below a confidence bar go to a person, how much does the
    model handle alone, and how often is it wrong on those?"""
    out = []
    for bar in [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
        auto = [r for r in rows if r["pred"] and (r["pred"]["confidence"] or 0) >= bar]
        wrong = [r for r in auto if r["pred"]["specialty"] != r["true"]["specialty"]
                 or r["pred"]["urgency"] != r["true"]["urgency"]]
        missed_urgent = [r for r in auto if r["true"]["urgency"] == "urgent"
                         and r["pred"]["urgency"] != "urgent"]
        out.append({"bar": bar, "handled_alone": round(len(auto) / len(rows), 3),
                    "error_rate_when_alone": round(len(wrong) / len(auto), 3) if auto else None,
                    "urgent_missed_alone": len(missed_urgent)})
    return out


def show(name, s):
    print(f"\n== {name}  (n={s['n']}, reply errors={s['errors']})")
    print(f"specialty {s['specialty_accuracy']:.0%}  urgency {s['urgency_accuracy']:.0%}  "
          f"both {s['both_correct']:.0%}  hard cases both {s['hard_both_correct']:.0%}")
    print(f"urgent recall {s['urgency']['urgent']['recall']:.0%}  "
          f"urgent missed {s['urgent_missed']}  urgent -> routine {s['urgent_sent_to_routine']}  "
          f"triage cost {s['triage_cost']}  median {s['median_seconds']}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="v1")
    ap.add_argument("--set", default="dev")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--compare", nargs="+")
    a = ap.parse_args()

    if a.compare:
        for f in a.compare:
            show(Path(f).stem, json.loads(Path(f).read_text())["score"])
        return

    cases = load(a.set)[: a.limit]
    t = time.time()
    rows = run(cases, a.prompt, a.thinking, a.workers)
    s = score(rows)
    model = os.environ.get("LLM_MODEL", "gemma4:e4b").replace(":", "-").replace("/", "-")
    name = f"{a.set}-{model}-{a.prompt}" + ("-thinking" if a.thinking else "")
    out = {"name": name, "model": model, "prompt": a.prompt, "thinking": a.thinking,
           "wall_seconds": round(time.time() - t, 1), "score": s,
           "review_sweep": review_sweep(rows), "rows": rows}
    (HERE / "results" / f"{name}.json").write_text(json.dumps(out, indent=2))
    show(name, s)
    print(json.dumps(s["urgency_confusion"], indent=2))


if __name__ == "__main__":
    main()
