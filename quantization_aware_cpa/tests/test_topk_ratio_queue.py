import pytest

from qacpa import queue_topk_ratios_same80 as queue


@pytest.mark.parametrize("failure", [None, "exit", "missing_marker"])
def test_queue_is_sequential_stops_on_failure_and_requires_markers(tmp_path, monkeypatch, failure):
    original_project = queue.PROJECT
    monkeypatch.setattr(queue, "PROJECT", tmp_path)
    for percent in (30, 20, 10):
        name = f"configs/cpa_fia_topk{percent}_same80_B16.json"
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((original_project / name).read_bytes())
    events = []

    class Process:
        def __init__(self, command, **kwargs):
            self.percent = int(command[-3].split("topk")[-1].split("_")[0])
            self.pid = 1000 + self.percent
            assert not events or events[-1][0] == "wait"
            events.append(("start", self.percent))

        def wait(self):
            events.append(("wait", self.percent))
            folder = tmp_path / f"results/cpa_fia_topk{self.percent}_same80_B16"
            state = "running" if failure == "missing_marker" and self.percent == 20 else "complete"
            queue.write_json(folder / "execution_status.json", {"state": state})
            return 7 if failure == "exit" and self.percent == 20 else 0

    monkeypatch.setattr(queue.subprocess, "Popen", Process)
    audits, reports = [], []
    monkeypatch.setattr(queue, "input_audit", lambda output: audits.append(output))
    monkeypatch.setattr(queue, "comparison_report", lambda output: reports.append(output))
    output = tmp_path / "queue"
    if failure:
        with pytest.raises(RuntimeError):
            queue.run_queue("run", output)
        assert events == [("start", 30), ("wait", 30), ("start", 20), ("wait", 20)]
        assert not audits and not reports
        assert queue.read_json(output / "queue_status.json")["state"] == "failed"
    else:
        queue.run_queue("run", output)
        assert events == [(action, p) for p in (30, 20, 10) for action in ("start", "wait")]
        assert audits == reports == [output]
        assert queue.read_json(output / "queue_status.json")["completed_keep_percents"] == [30, 20, 10]
