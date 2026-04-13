#!/bin/bash
# PostToolUse: valida estado después de cada agente
STATE_FILE="outputs/latest/state_current.json"
if [ -f "$STATE_FILE" ]; then
  python src/utils/state_validator.py "$STATE_FILE"
fi
