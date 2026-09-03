class QACPAError(RuntimeError):
    """Base exception for qacpa."""


class GateFailure(QACPAError):
    """A blocking milestone gate failed."""
