import json
from pathlib import Path
from qacpa import signed_cell_rank_evaluation as e
path=Path('quantization_aware_cpa/results/signed_int4_baseline_B16_T10/naive/fia_initial.json')
initial=json.loads(path.read_text(encoding='utf-8'))
e._validate_initialization(initial, 'existing_naive')
print('existing_naive_initial=VALID')
print('shape=', initial['shape'])
print('seed=', initial['seed'])
print('runtime=', json.dumps(initial['runtime'], sort_keys=True))
