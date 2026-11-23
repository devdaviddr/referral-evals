#!/bin/sh
# Follow-up: does prompt v4 fix the time-critical gap without breaking anything?
set -e
cd "$(dirname "$0")"
export LLM_MODEL=gemma4:12b
python3 evaluate.py --prompt v3 --set dev2 --workers 1
python3 evaluate.py --prompt v4 --set dev2 --workers 1
python3 evaluate.py --prompt v4 --set dev --workers 1
