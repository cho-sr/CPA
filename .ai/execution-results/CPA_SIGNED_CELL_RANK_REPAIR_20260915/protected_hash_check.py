import hashlib, json
from pathlib import Path
parent = json.loads(Path('.ai/execution-results/CPA_SIGNED_CELL_RANK_IMPLEMENTATION_20260914/results.json').read_text(encoding='utf-8'))
expected = parent['protected_source_hashes_current']
actual = {}
for relative in expected:
    data = Path(relative).read_bytes()
    actual[relative] = hashlib.sha256(data).hexdigest()
print(json.dumps({'expected': expected, 'actual': actual, 'match': actual == expected}, indent=2))
Path('.ai/execution-results/CPA_SIGNED_CELL_RANK_REPAIR_20260915/protected_hashes.json').write_text(json.dumps({'expected':expected,'actual':actual,'match':actual==expected},indent=2)+'\n',encoding='utf-8')
raise SystemExit(0 if actual == expected else 1)
