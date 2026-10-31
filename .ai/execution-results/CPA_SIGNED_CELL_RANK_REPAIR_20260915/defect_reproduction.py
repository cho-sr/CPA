import torch
from qacpa import signed_cell_rank as core
from qacpa import signed_cell_rank_evaluation as evaluation
from qacpa import signed_cell_rank_runner as runner

print('R1 canonical hash encoding probe')
v = torch.tensor([[1.0, -2.0], [3.5, 4.0]], dtype=torch.float32)
print('core_tensor_sha256=', core.tensor_sha256(v))
print('upstream_tensor_hash=', runner.upstream_naive._tensor_hash(v))
print('encodings_equal=', core.tensor_sha256(v) == runner.upstream_naive._tensor_hash(v))

print('R2 pre-fix compact layout probe')
axes = torch.empty((4, 9, 1))
try:
    for truth_index in range(16):
        row, offset = truth_index // 4, (truth_index % 4) * 3
        axes[row, offset + 2, 0] = truth_index
except IndexError as error:
    print('old_compact_shape=(4,9) failure=', type(error).__name__, str(error))
print('repaired_compact_shape=(4,12) required_max_column=', 11)

print('R3 movement lag probe')
v0 = torch.zeros((2, 2))
v1 = torch.ones((2, 2))
v2 = torch.full((2, 2), 2.0)
v3 = torch.full((2, 2), 3.0)
print('step3_expected_consecutive=', core._movement(v3, v2))
print('step3_old_lagging_previous=', core._movement(v3, v1))
print('movement_records_differ=', core._movement(v3, v2) != core._movement(v3, v1))

print('R4 stage status contract')
print('completed_attack_pending_eval=', {'attack_status': 'COMPLETED', 'evaluation_status': 'NOT_RUN', 'full_experiment_status': 'NOT_RUN'})
print('failed_fia=', {'attack_status': 'FAILED', 'cpa_status': 'COMPLETED', 'fia_status': 'FAILED', 'evaluation_status': 'NOT_RUN'})

print('R5 exact settings guard keys')
print('cpa_keys=', sorted(evaluation.CPA_EXECUTED_SETTINGS))
print('fia_keys=', sorted(evaluation.FIA_EXECUTED_SETTINGS))
