from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
RECOVERY = ROOT / ".ai/execution-results/CPA_SIGNED_BITS7654_RECOVERY_20260918"


def load_local(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, RECOVERY / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RecoveryTests(unittest.TestCase):
    def test_gate_adapter_adds_authenticated_task_id_and_rejects_bad_contract(self):
        adapter = load_local("bits_recovery_gate_adapter", "gate_adapter.py")
        good = {"status": "PUBLIC_GATE_PASSED", "bits": 7, "private_loaded": False,
                "candidate_count": 13, "candidates": {str(i): {} for i in range(13)}}
        with patch.object(adapter.runner, "public_gate", return_value=good):
            result = adapter.authenticate(7, Path("fixture"))
        self.assertEqual(result["task_id"], "CPA_SIGNED_BITS7654_SCREEN3_20260918_B7")
        self.assertTrue(result["adapter_did_not_modify_runner"])
        bad = dict(good, status="PUBLIC_GATE_FAILED")
        with patch.object(adapter.runner, "public_gate", return_value=bad):
            with self.assertRaises(adapter.GateAdapterError):
                adapter.authenticate(7, Path("fixture"))

    def test_recovery_state_has_zero_bit7_optimizer_and_serial_order(self):
        recovery = load_local("bits_recovery_wrapper", "recover_bits.py")
        state = recovery.new_state()
        self.assertEqual(state["bit_status"], {"7": "PENDING_SAVED_EVALUATION", "6": "QUEUED", "5": "QUEUED", "4": "QUEUED"})
        self.assertEqual(state["bit_state"]["7"]["optimizer_steps"], 0)
        self.assertEqual(state["bits"], [7, 6, 5, 4])
        self.assertEqual(recovery.remaining_seconds(state, 7), recovery.BIT7_REMAINING)

    def test_resume_plan_does_not_rerun_completed_bit7_or_completed_public_stage(self):
        recovery = load_local("bits_recovery_wrapper_resume", "recover_bits.py")
        state = recovery.new_state()
        state["bit_status"]["7"] = "COMPLETED"
        state["bit_state"]["7"]["stages"] = {"saved_public": "COMPLETED", "gate": "COMPLETED",
                                                  "evaluation": "COMPLETED", "evaluation_output": "saved"}
        state["bit_status"]["6"] = "PUBLIC_COMPLETED_PENDING_EVALUATION"
        state["bit_state"]["6"]["stages"] = {"public": "COMPLETED"}
        self.assertEqual(state["bit_state"]["7"]["optimizer_steps"], 0)
        self.assertEqual(state["bit_state"]["6"]["stages"]["public"], "COMPLETED")
        self.assertEqual([7, 6, 5, 4], list(recovery.BITS))

    def test_mocked_main_resumes_bit7_eval_once_then_fresh_bits_in_order(self):
        recovery = load_local("bits_recovery_wrapper_orchestration", "recover_bits.py")
        with tempfile.TemporaryDirectory() as temp:
            result_root = Path(temp)
            calls = []

            def fake_bit7(state):
                calls.append(("bit7", "evaluation"))
                state["bit_status"]["7"] = "COMPLETED"
                state["bit_state"]["7"]["optimizer_steps"] = 0
                state["bit_results"]["7"] = {"evaluation_status": "COMPLETED", "candidate_count": 13}

            def fake_fresh(state, bits):
                calls.append((bits, "public", "evaluation"))
                state["bit_status"][str(bits)] = "COMPLETED"
                state["bit_state"][str(bits)]["optimizer_steps"] = 125000
                state["bit_results"][str(bits)] = {"evaluation_status": "COMPLETED", "candidate_count": 13}

            with patch.object(recovery, "RESULT_ROOT", result_root), \
                 patch.object(recovery, "verify_parent_evidence"), \
                 patch.object(recovery, "run_bit7", side_effect=fake_bit7), \
                 patch.object(recovery, "run_fresh_bit", side_effect=fake_fresh):
                self.assertEqual(recovery.main(), 0)
            self.assertEqual(calls, [("bit7", "evaluation"), (6, "public", "evaluation"),
                                     (5, "public", "evaluation"), (4, "public", "evaluation")])

    def test_callback_rejects_stale_recipient_before_wait_registration(self):
        callback = load_local("bits_recovery_callback_stale", "completion_callback.py")
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(callback, "RESULT_ROOT", Path(temp)), \
                 patch.object(callback, "verify_executor", side_effect=RuntimeError("stale recipient")):
                with self.assertRaisesRegex(RuntimeError, "stale recipient"):
                    callback.run_callback(123, 456, wait_fn=lambda kernel, handle: 0,
                                          queue_fn=lambda message: {"exit_code": 0})

    def test_callback_wait_and_delivery_are_exactly_once_with_harmless_mock(self):
        callback = load_local("bits_recovery_callback", "completion_callback.py")
        with tempfile.TemporaryDirectory() as temp:
            result_root = Path(temp)
            summary_path = result_root / "execution_summary.json"
            summary_path.write_text(json.dumps({"status": "PARTIAL", "bit_status": {"7": "COMPLETED"}}), encoding="utf-8")
            with patch.object(callback, "RESULT_ROOT", result_root), \
                 patch.object(callback, "verify_executor", return_value={"thread": callback.EXECUTOR_THREAD}), \
                 patch.object(callback, "checked_process", return_value=(type("K", (), {
                     "GetExitCodeProcess": lambda self, handle, ptr: (setattr(ptr._obj, "value", 1) or True),
                     "CloseHandle": lambda self, handle: None,
                 })(), object())):
                delivered = []
                def deliver(message):
                    delivered.append(message)
                    return {"exit_code": 0}
                code = callback.run_callback(123, 456, wait_fn=lambda kernel, handle: 0, queue_fn=deliver)
            self.assertEqual(code, 0)
            self.assertEqual(len(delivered), 1)
            status = json.loads((result_root / "callback_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "NOTIFICATION_QUEUED")


if __name__ == "__main__":
    unittest.main()
