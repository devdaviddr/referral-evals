"""Offline tests for reply parsing and scoring. No model needed.

    python3 -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bootstrap import ci  # noqa: E402
from classify import parse  # noqa: E402
from evaluate import prf, review_sweep, score  # noqa: E402


def row(true_u, pred_u, true_s="cardiology", pred_s="cardiology", conf=0.9, hard=False, rid="x"):
    pred = None if pred_u is None else {"specialty": pred_s, "urgency": pred_u,
                                        "confidence": conf, "reason": ""}
    return {"id": rid, "hard": hard, "true": {"specialty": true_s, "urgency": true_u},
            "pred": pred, "error": None, "seconds": 1.0}


class ParseTests(unittest.TestCase):
    def test_plain_json(self):
        out = parse('{"specialty": "Cardiology", "urgency": "Urgent", "confidence": 0.8, "reason": "r"}')
        self.assertEqual(out["specialty"], "cardiology")
        self.assertEqual(out["urgency"], "urgent")
        self.assertEqual(out["confidence"], 0.8)

    def test_json_inside_prose_and_think_block(self):
        reply = '<think>{"specialty": "neurology"}</think>Sure: {"specialty": "respiratory", "urgency": "routine"}'
        out = parse(reply)
        self.assertEqual(out["specialty"], "respiratory")
        self.assertIsNone(out["confidence"])

    def test_rejects_unknown_labels(self):
        with self.assertRaises(ValueError):
            parse('{"specialty": "oncology", "urgency": "urgent"}')
        with self.assertRaises(ValueError):
            parse('{"specialty": "cardiology", "urgency": "asap"}')

    def test_rejects_empty_reply(self):
        with self.assertRaises(ValueError):
            parse("")


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            row("urgent", "urgent"),
            row("urgent", "routine"),          # worst mistake: cost 10
            row("urgent", "semi-urgent"),      # cost 3
            row("semi-urgent", "urgent"),      # over-call: cost 1
            row("routine", "routine", true_s="dermatology", pred_s="cardiology", hard=True),
            row("routine", None),              # bad reply: counted as a miss, cost 10
        ]

    def test_triage_cost_weights_mistakes_by_harm(self):
        self.assertEqual(score(self.rows)["triage_cost"], 10 + 3 + 1 + 10)

    def test_urgent_counts(self):
        s = score(self.rows)
        self.assertEqual(s["urgent_missed"], 2)
        self.assertEqual(s["urgent_sent_to_routine"], 1)
        self.assertEqual(s["errors"], 1)

    def test_accuracy_counts_bad_replies_as_wrong(self):
        s = score(self.rows)
        self.assertAlmostEqual(s["urgency_accuracy"], round(2 / 6, 3))
        self.assertAlmostEqual(s["both_correct"], round(1 / 6, 3))
        self.assertEqual(s["hard_both_correct"], 0.0)

    def test_precision_and_recall(self):
        p = prf(self.rows, "urgency", "urgent")
        self.assertEqual(p["support"], 3)
        self.assertAlmostEqual(p["recall"], round(1 / 3, 3))
        self.assertAlmostEqual(p["precision"], 0.5)

    def test_always_routine_baseline(self):
        rows = [row("urgent", "routine")] * 2 + [row("routine", "routine")] * 2
        s = score(rows)
        self.assertEqual(s["urgency_accuracy"], 0.5)
        self.assertEqual(s["urgency"]["urgent"]["recall"], 0.0)


class SweepTests(unittest.TestCase):
    def test_high_bar_sends_low_confidence_letters_to_a_person(self):
        rows = [row("urgent", "urgent", conf=0.99), row("urgent", "routine", conf=0.6)]
        sweep = {s["bar"]: s for s in review_sweep(rows)}
        self.assertEqual(sweep[0.0]["handled_alone"], 1.0)
        self.assertEqual(sweep[0.0]["urgent_missed_alone"], 1)
        self.assertEqual(sweep[0.95]["handled_alone"], 0.5)
        self.assertEqual(sweep[0.95]["urgent_missed_alone"], 0)


class BootstrapTests(unittest.TestCase):
    def test_interval_bounds(self):
        lo, hi = ci([i / 100 for i in range(101)])
        self.assertLessEqual(lo, 0.03)
        self.assertGreaterEqual(hi, 0.97)


if __name__ == "__main__":
    unittest.main()
