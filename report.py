"""Rebuild the published results tables from the saved runs in results/.

Every number in the article and the README comes from here, so they can be
checked without calling a model.

    python3 report.py
"""
import json
from pathlib import Path

RESULTS = Path(__file__).parent / "results"

MAIN = [  # (label, run file) for the main comparison on the 120 tuning letters
    ("v1, e4b", "dev-gemma4-e4b-v1"),
    ("v2, e4b", "dev-gemma4-e4b-v2"),
    ("v3, e4b", "dev-gemma4-e4b-v3"),
    ("v3, e4b with reasoning", "dev-gemma4-e4b-v3-thinking"),
    ("v3, 12b", "dev-gemma4-12b-v3"),
]
HELD_OUT = [("v3, 12b on 60 held-out letters", "test-gemma4-12b-v3")]
FOLLOW_UP = [  # (letters, v3 run, v4 run)
    ("30 new tuning letters", "dev2-gemma4-12b-v3", "dev2-gemma4-12b-v4"),
    ("Original 120 letters", "dev-gemma4-12b-v3", "dev-gemma4-12b-v4"),
    ("30 new held-out letters", "test2-gemma4-12b-v3", "test2-gemma4-12b-v4"),
]


def load(name):
    return json.loads((RESULTS / f"{name}.json").read_text())["score"]


def caught(s):
    u = s["urgency"]["urgent"]
    return f"{u['support'] - s['urgent_missed']} of {u['support']}"


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + " --- |" * len(header))
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print()


def main():
    print("Main comparison, 120 tuning letters\n")
    table(["Setup", "Both labels right", "Urgent caught", "Triage cost", "Seconds per letter"],
          [(label, f"{s['both_correct']:.0%}", caught(s), s["triage_cost"], s["median_seconds"])
           for label, s in ((l, load(n)) for l, n in MAIN + HELD_OUT)])

    print("Follow-up: prompt v3 against v4, 12b model\n")
    rows = []
    for letters, a, b in FOLLOW_UP:
        sa, sb = load(a), load(b)
        rows.append((letters, caught(sa), caught(sb), f"{sa['both_correct']:.0%}",
                     f"{sb['both_correct']:.0%}", sa["triage_cost"], sb["triage_cost"]))
    table(["Letters", "Urgent caught, v3", "Urgent caught, v4", "Both right, v3",
           "Both right, v4", "Cost, v3", "Cost, v4"], rows)

    lc = RESULTS / "labelcheck-qwen3.5-9b.json"
    adj = RESULTS / "labelcheck-adjudication.json"
    if lc.exists() and adj.exists():
        s = json.loads(lc.read_text())["summary"]
        verdicts = [a["verdict"] for a in json.loads(adj.read_text())]
        print("Label check (Qwen 3.5 9b, blind)\n")
        table(["Labelled", "Specialty agrees", "Urgency agrees", "Key clearly wrong",
               "Borderline", "Second model wrong"],
              [(f"{s['labelled']} of {s['of']}", s["specialty_agree"], s["urgency_agree"],
                verdicts.count("key_wrong"), verdicts.count("ambiguous"),
                verdicts.count("key_right"))])


if __name__ == "__main__":
    main()
