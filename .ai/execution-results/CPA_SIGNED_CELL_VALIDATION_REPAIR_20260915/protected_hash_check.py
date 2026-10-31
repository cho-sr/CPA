import hashlib, json
from pathlib import Path
parent = json.loads(Path('.ai/execution-results/CPA_SIGNED_CELL_RANK_IMPLEMENTATION_20260914/results.json').read_text(encoding='utf-8'))
expected = parent['protected_source_hashes_current']
actual = {key: hashlib.sha256(Path(key).read_bytes()).hexdigest() for key in expected}
core_path = 'quantization_aware_cpa/qacpa/signed_cell_rank.py'
core_hash = hashlib.sha256(Path(core_path).read_bytes()).hexdigest()
value = {'protected_expected': expected, 'protected_actual': actual,
         'protected_match': actual == expected, 'signed_cell_rank_sha256': core_hash}
Path('.ai/execution-results/CPA_SIGNED_CELL_VALIDATION_REPAIR_20260915/protected_hashes.json').write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'protected_match': actual == expected, 'signed_cell_rank_sha256': core_hash}, indent=2))
raise SystemExit(0 if actual == expected else 1)
