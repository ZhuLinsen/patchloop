#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
python3 review.py >> "logs/$(date +%Y-%m-%d).log" 2>&1
