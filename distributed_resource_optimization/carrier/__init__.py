from .core import Carrier, EventWithValue
from .simple import (
    ActorContainer,
    SimpleCarrier,
    cid,
    start_coordinated_optimization,
    start_distributed_optimization,
)

__all__ = [
    "Carrier",
    "EventWithValue",
    "ActorContainer",
    "SimpleCarrier",
    "cid",
    "start_distributed_optimization",
    "start_coordinated_optimization",
]

import sys as _sys
from importlib import import_module as _import_module
from importlib.util import find_spec as _find_spec

if _find_spec("mango") is not None:
    _sys.modules[f"{__name__}.mango"] = _import_module(f"{__name__}.mango_carrier")
