# Recreate the three reviewed pre-fix validation holes with tiny values only.
from qacpa import signed_cell_rank_evaluation as evaluation
import torch

print('V1 missing-field equality hole')
left, right = {}, {}
print('old_missing_runtime_equal=', left.get('runtime', {}).get('threads') == right.get('runtime', {}).get('threads'))
print('old_bool_int_equal=', True == 1)

print('V2 metadata declared-arm hole')
registration, result = {'rank': 16, 'ap_iterations': 0}, {'rank': 16, 'ap_iterations': 5}
print('old_registration_result_partial_check=', registration.get('rank') == result.get('rank'))
print('old_ap0_ap5_declared_contract=', None)

print('V3 zero-reference hole')
try:
    _ = 0.0 / 0.0
except ZeroDivisionError as error:
    print('old_zero_reference=', type(error).__name__)
fixed = evaluation.update_error_metrics(
    torch.zeros((1, 2), dtype=torch.float32), torch.zeros((1, 2), dtype=torch.float32),
    arm='probe', estimate_input_sha256='a' * 64,
    delta_convention=evaluation.RAW_DELTA_CONVENTION)
print('repaired_zero_status=', fixed['status'])
print('repaired_zero_relative=', fixed['relative_l2_error'])
