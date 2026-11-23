# referral-evals

A small LLM classifier that sorts GP referral letters by specialty and urgency, and the eval that tests it.

Companion code for the article [Classifying referrals with an LLM, and testing whether it works](https://www.danielruffolo.net/posts/2026-11-22-classifying-referrals-with-an-llm-evals/).

All 240 letters are synthetic. No real patient data is used.

## What it does

Each letter gets:

- a specialty: cardiology, orthopaedics, neurology, gastroenterology, dermatology or respiratory
- an urgency, using the Australian outpatient referral categories: urgent (within 30 days), semi-urgent (within 90 days) or routine (within 12 months)
- a confidence and a one-line reason

## Run it

Python 3 standard library only. It talks to any OpenAI-compatible endpoint. The default is Ollama on this machine.

```bash
ollama pull gemma4:e4b
echo "58yo smoker, 3 weeks of chest tightness walking uphill." | python3 classify.py

python3 evaluate.py --prompt v1 --set dev --workers 1
python3 evaluate.py --prompt v3 --set dev --workers 1 --thinking
LLM_MODEL=gemma4:12b python3 evaluate.py --prompt v3 --set test --workers 1
python3 evaluate.py --compare results/dev-gemma4-e4b-v1.json results/dev-gemma4-e4b-v3.json
```

`./run_all.sh` runs every dev-set eval from the article.

Settings: `LLM_BASE_URL` (default `http://127.0.0.1:11434/v1`), `LLM_MODEL` (default `gemma4:e4b`), `LLM_API_KEY` (optional).

## Files

- `classify.py`: sends one letter to the model and checks the reply
- `evaluate.py`: runs a labelled set, scores it and saves the run to `results/`
- `prompts/`: the four prompt versions from the article, plus the label-check prompt
- `data/dev.jsonl`: 120 letters for tuning
- `data/test.jsonl`: 60 held-out letters, run once at the end
- `data/dev2.jsonl`, `data/test2.jsonl`: 30 + 30 follow-up letters aimed at time-critical surgical cases and their look-alikes
- `run_followup.sh`: the follow-up runs that test prompt v4
- `bootstrap.py`: resamples the letters to show how much of a difference between runs could be chance
- `labelcheck.py`: a second model (Qwen 3.5 9b) labels every letter blind, as a check on the answer key
- `results/`: the runs behind the numbers in the article, including the label check and the review of each disagreement (`labelcheck-adjudication.json`)

## What gets measured

Accuracy for specialty, urgency and both together; precision and recall per label; how many urgent letters were missed or sent to routine; a triage cost that weights dangerous mistakes heavily; time per letter; and a sweep showing what happens if letters below a confidence bar go to a person.

## Limits

The letters are synthetic and no clinician reviewed the labels; a second model checked them instead. There are 24 urgent letters in the dev set and 12 in the test set, so small differences can be chance. Only Gemma 4 was tested.

## Licence

MIT
