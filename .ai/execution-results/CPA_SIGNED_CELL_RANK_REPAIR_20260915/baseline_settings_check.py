import json
from pathlib import Path
from qacpa import signed_cell_rank_evaluation as e
root=Path('quantization_aware_cpa/results/signed_int4_baseline_B16_T10/naive')
for kind, expected in [('cpa',e.CPA_EXECUTED_SETTINGS),('fia',e.FIA_EXECUTED_SETTINGS)]:
    result=json.loads((root/f'{kind}_result.json').read_text(encoding='utf-8'))
    manifest=json.loads((root/kind/f'{kind}_manifest.json').read_text(encoding='utf-8'))
    print(kind, 'result_settings_equal=', result.get('settings') == expected)
    print(kind, 'manifest_settings_equal=', manifest.get('fingerprint',{}).get('settings') == expected)
    print(kind, 'result_n_iter=', result.get('settings',{}).get('n_iter'))
    print(kind, 'result_lr=', result.get('settings',{}).get('lr'))
