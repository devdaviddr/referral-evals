#!/bin/sh
# Every dev-set run for the article, one letter at a time so timings are fair.
set -e
cd "$(dirname "$0")"
for p in v1 v2 v3; do LLM_MODEL=gemma4:e4b python3 evaluate.py --prompt $p --set dev --workers 1; done
LLM_MODEL=gemma4:e4b python3 evaluate.py --prompt v3 --set dev --workers 1 --thinking
LLM_MODEL=gemma4:12b python3 evaluate.py --prompt v3 --set dev --workers 1
