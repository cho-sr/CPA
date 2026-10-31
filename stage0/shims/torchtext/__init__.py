"""Import-time shim that shadows the unusable ``torchtext`` in this env.

Why this exists
---------------
Upstream ``official_cpa/src/datasets.py`` imports torchtext at module scope for
the (out-of-scope) IMDB text pipeline.  The env's real ``torchtext`` 0.18 ships
a compiled ``libtorchtext.pyd`` built against an older libtorch and raises
``OSError`` on import under torch 2.13.

A runtime ``sys.modules`` stub is not enough: Windows ``spawn`` re-executes the
DataLoader worker's ``__main__`` from its file path, so the child imports
upstream ``datasets.py`` in a fresh interpreter with no stub installed.
``sys.path`` *is* inherited by spawned children, so shadowing the package on
``sys.path`` fixes parent and workers alike.

``stage0/run_cpa.py`` prepends ``stage0/shims`` to ``sys.path``.

Anything that actually tries to *use* torchtext raises, so this cannot silently
change results for a text experiment.
"""

__version__ = "0.0.0+stage0-shim"


def _unavailable(*args, **kwargs):
    raise RuntimeError(
        "torchtext is shimmed out for Stage 0 (image pipelines only). "
        "See stage0/shims/torchtext/__init__.py"
    )
