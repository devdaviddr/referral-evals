#!/bin/sh
# Re-run every eval behind the article, in order. Needs Ollama with
# gemma4:e4b and gemma4:12b pulled. About 2.5 hours on a 16 GB Mac mini,
# most of it the reasoning run; skip that line to save about 50 minutes.
set -e
cd "$(dirname "$0")"

# 1. Prompt versions on the 120 tuning letters
for p in v1 v2 v3; do LLM_MODEL=gemma4:e4b python3 evaluate.py --prompt $p --set dev --workers 1; done

# 2. Reasoning on, and the bigger model
LLM_MODEL=gemma4:e4b python3 evaluate.py --prompt v3 --set dev --workers 1 --thinking
LLM_MODEL=gemma4:12b python3 evaluate.py --prompt v3 --set dev --workers 1

# 3. Held-out test, run once with the chosen setup
LLM_MODEL=gemma4:12b python3 evaluate.py --prompt v3 --set test --workers 1

# 4. Follow-up: does prompt v4 fix time-critical surgical cases without regressions?
./run_followup.sh
LLM_MODEL=gemma4:12b python3 evaluate.py --prompt v3 --set test2 --workers 1
LLM_MODEL=gemma4:12b python3 evaluate.py --prompt v4 --set test2 --workers 1

# 5. How much of each difference could be chance
python3 bootstrap.py results/dev-gemma4-e4b-v1.json results/dev-gemma4-e4b-v2.json \
  results/dev-gemma4-e4b-v3.json results/dev-gemma4-e4b-v3-thinking.json results/dev-gemma4-12b-v3.json
