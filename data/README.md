# Data

240 synthetic GP-to-specialist referral letters, written for this eval. No real patients or real clinicians: every name, age and detail is invented.

## Files

| File | Letters | Urgent | Semi-urgent | Routine | Hard | Used for |
| --- | --- | --- | --- | --- | --- | --- |
| `dev.jsonl` | 120 | 24 | 36 | 60 | 24 | Tuning prompts and choosing a model |
| `test.jsonl` | 60 | 12 | 18 | 30 | 13 | Held out, run once with the chosen setup |
| `dev2.jsonl` | 30 | 13 | 6 | 11 | 9 | Follow-up tuning: time-critical surgical cases |
| `test2.jsonl` | 30 | 13 | 5 | 12 | 9 | Follow-up held out |

`dev` and `test` follow a realistic mix (about 20% urgent, 30% semi-urgent, 50% routine) with each specialty equally represented. `dev2` and `test2` are deliberately skewed: roughly a third are urgent because a decision about surgery can't wait, and a third are non-urgent look-alikes (chronic tendon pain, healed ruptures, stable tears) that a careless fix would over-call.

## Format

One JSON object per line:

```json
{
  "id": "test-011",
  "text": "Re: Daniel Ferreira, 39 ...",
  "specialty": "orthopaedics",
  "urgency": "urgent",
  "hard": true,
  "note": "Acute complete Achilles rupture where the management decision is time-critical."
}
```

| Field | Meaning |
| --- | --- |
| `id` | Unique id, prefixed with the file name |
| `text` | The referral letter, 60 to 150 words |
| `specialty` | cardiology, orthopaedics, neurology, gastroenterology, dermatology or respiratory |
| `urgency` | urgent (within 30 days), semi-urgent (within 90 days) or routine (within 12 months), following the Australian outpatient referral categories |
| `hard` | Deliberately difficult (see below) |
| `note` | One sentence explaining the label |

## Hard cases

About one letter in five is marked `hard`:

- vague letters with very little detail
- two conditions, where the referral is for the less obvious one
- a red flag buried in an otherwise routine letter
- symptoms that sit between two specialties, such as chest pain that might be reflux
- a GP asking for an urgency the clinical details don't support, in either direction

## How the labels were made and checked

A different model from the one under test (Claude) wrote the letters and their labels, so the classifier was never marked against its own reading.

No clinician reviewed the labels. As a check, Qwen 3.5 9b labelled the 180 `dev` and `test` letters blind. It agreed on specialty for 177 and on urgency for 137. A review of the 43 disagreements found no label that was clearly wrong, 14 borderline cases where either label is defensible, and 29 where Qwen had missed something. The borderline cases are all semi-urgent in the key and routine for Qwen, so the key leans slightly cautious at that boundary. Details are in `results/labelcheck-adjudication.json`.
