"""Second-opinion check on the answer key.

A model from a different family labels the 180 dev and test letters blind
(it never sees the key). Then we count agreement and list disagreements
for review. Reasoning is off: with it on, Qwen 3.5 9b ran past 3,700
tokens per letter without answering.

    LLM_MODEL=qwen3.5:9b python3 labelcheck.py
"""
import json
import os
import time
from pathlib import Path

from classify import classify

HERE = Path(__file__).parent
OUT = HERE / "results" / f"labelcheck-{os.environ.get('LLM_MODEL', 'model').replace(':', '-')}.json"


def main():
    done = {r["id"]: r for r in json.loads(OUT.read_text())["rows"]} if OUT.exists() else {}
    rows = []
    for name in ("dev", "test"):
        for line in (HERE / "data" / f"{name}.jsonl").open():
            case = json.loads(line)
            if case["id"] in done and done[case["id"]]["second"]:
                rows.append(done[case["id"]])
                continue
            t = time.time()
            try:
                second = classify(case["text"], prompt="labeller", thinking=False)
            except Exception as e:
                second = None
                print(case["id"], "error", e)
            rows.append({"id": case["id"], "hard": case["hard"],
                         "key": {"specialty": case["specialty"], "urgency": case["urgency"]},
                         "second": second, "seconds": round(time.time() - t, 1)})
            OUT.write_text(json.dumps({"rows": rows}, indent=2))  # resumable
            print(case["id"], second and (second["specialty"], second["urgency"]), flush=True)

    ok = [r for r in rows if r["second"]]
    agree = lambda f: sum(r["second"][f] == r["key"][f] for r in ok)
    both = sum(r["second"]["specialty"] == r["key"]["specialty"]
               and r["second"]["urgency"] == r["key"]["urgency"] for r in ok)
    summary = {"labelled": len(ok), "of": len(rows),
               "specialty_agree": agree("specialty"), "urgency_agree": agree("urgency"),
               "both_agree": both,
               "disagreements": [r["id"] for r in ok
                                 if r["second"]["specialty"] != r["key"]["specialty"]
                                 or r["second"]["urgency"] != r["key"]["urgency"]]}
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
