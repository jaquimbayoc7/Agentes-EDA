#!/bin/bash
# PreCommit: verifica que no haya encoding inconsistente
echo "Checking encoding consistency..."
python -c "
import json, sys
from pathlib import Path
state_files = list(Path('outputs').rglob('state_final.json'))
for sf in state_files[-1:]:
    data = json.loads(sf.read_text())
    log = data.get('encoding_log', {})
    for col, info in log.items():
        if 'encoding_final' not in info:
            print(f'WARNING: {col} missing encoding_final in {sf}')
            sys.exit(1)
print('Encoding check: OK')
"
