"""
tensorflow/__init__.py — Compatibility shim for lstm_ids_project
================================================================
This module stands in for the real `tensorflow` package, routing all
`import tensorflow as tf` and `from tensorflow.keras import ...`
calls to standalone Keras 3 (running on the PyTorch backend).

WHY THIS EXISTS
---------------
The host CPU lacks AVX/AVX2 instructions. Every prebuilt TensorFlow
wheel since TF 1.6 is compiled with AVX, causing an immediate
"Illegal hardware instruction" (SIGILL) crash on import. This shim
lets the project run without AVX by using Keras 3 on top of PyTorch,
which ships a CPU build that is SSE4-only compatible.

WHAT IT PROVIDES
----------------
  tf.keras                          → keras (Keras 3 API)
  tf.keras.Sequential               → keras.Sequential
  tf.keras.Model                    → keras.Model
  tf.keras.metrics.Precision/Recall → keras.metrics.*
  tf.keras.optimizers.Adam          → keras.optimizers.Adam
  tf.keras.callbacks.*              → keras.callbacks.*
  tf.keras.models.load_model        → keras.models.load_model
  tf.keras.backend.clear_session    → keras.backend.clear_session
  tf.random.set_seed(seed)          → torch.manual_seed + random seeds

USAGE
-----
No source changes required. Python's import system finds this
`tensorflow/` directory (at the project root) before searching
site-packages, so `import tensorflow as tf` silently picks this up.

SETUP (one-time, in your venv)
-------------------------------
    pip install "keras>=3.0" keras-tuner
    # PyTorch is already installed and working.
    # Set Keras backend via environment variable or keras.json:
    export KERAS_BACKEND=torch
"""

import os
import sys
import types
import random as _random

# ── 1. Force Keras to use the PyTorch backend ─────────────────────────────────
os.environ.setdefault("KERAS_BACKEND", "torch")

# ── 2. Import standalone Keras ─────────────────────────────────────────────────
try:
    import keras as _keras
except ImportError as exc:
    raise ImportError(
        "\n\n[tensorflow shim] Could not import 'keras'.\n"
        "Run:  pip install 'keras>=3.0'\n"
        "This project uses a TF-compatibility shim that delegates to\n"
        "standalone Keras 3 on the PyTorch backend (no AVX required).\n"
    ) from exc

# ── 3. Build the `tensorflow.keras` namespace ─────────────────────────────────
#      We expose everything Keras provides, plus add any missing compat aliases.

keras = _keras  # tf.keras → keras

# ── 4. Attach sub-namespaces the project code accesses ────────────────────────

# tf.keras.backend
if not hasattr(_keras, "backend"):
    _keras.backend = types.SimpleNamespace()
if not hasattr(_keras.backend, "clear_session"):
    # Keras 3 exposes this at keras.backend.clear_session
    try:
        from keras import backend as _kb
        _keras.backend = _kb
    except Exception:
        _keras.backend = types.SimpleNamespace(clear_session=lambda: None)

# tf.keras.models  (load_model etc.)
if not hasattr(_keras, "models"):
    _keras.models = types.SimpleNamespace(load_model=_keras.saving.load_model)


# ── 5. Seed helper ────────────────────────────────────────────────────────────
class _Random:
    """Provides tf.random.set_seed(seed) without TF."""
    @staticmethod
    def set_seed(seed: int) -> None:
        import numpy as np
        _random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass
        try:
            _keras.utils.set_random_seed(seed)
        except Exception:
            pass


random = _Random()


# ── 6. TensorFlow API stubs ───────────────────────────────────────────────────
#      Keras 3 internals import `tf.TensorShape`, `tf.DType`, etc. via its
#      LazyModule proxy.  Because our shim makes `tf.available == True`, those
#      lookups must succeed.  On the PyTorch backend the stubs are never
#      instantiated — they just need to exist so isinstance() returns False
#      and getattr() doesn't crash.

class TensorShape(tuple):
    """Minimal stand-in for ``tf.TensorShape``."""
    def __new__(cls, dims=None):
        if dims is None:
            return super().__new__(cls, ())
        if isinstance(dims, int):
            dims = (dims,)
        return super().__new__(cls, tuple(dims))

    def as_list(self):
        return list(self)

    @property
    def rank(self):
        return len(self)

    @property
    def _dims(self):
        return list(self) if len(self) else None


class DType:
    """Minimal stand-in for ``tf.DType``."""
    def __init__(self, name="float32"):
        self.name = str(name)

    def __repr__(self):
        return f"tf.{self.name}"

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, DType):
            return self.name == other.name
        return self.name == str(other)

    def __hash__(self):
        return hash(self.name)


class TypeSpec:
    """Minimal stand-in for ``tf.TypeSpec``."""
    pass


class TensorSpec(TypeSpec):
    """Minimal stand-in for ``tf.TensorSpec``."""
    def __init__(self, shape=None, dtype=None, name=None):
        self.shape = TensorShape(shape) if shape is not None else TensorShape()
        self.dtype = DType(dtype) if dtype is not None else DType("float32")
        self.name = name


class Tensor:
    """Placeholder so ``isinstance(x, tf.Tensor)`` returns False."""
    pass


class SparseTensor:
    """Placeholder so ``isinstance(x, tf.SparseTensor)`` returns False."""
    pass


class IndexedSlices:
    """Placeholder so ``isinstance(x, tf.IndexedSlices)`` returns False."""
    pass


class RaggedTensor:
    """Placeholder so ``isinstance(x, tf.RaggedTensor)`` returns False."""
    pass


class Variable:
    """Placeholder so ``isinstance(x, tf.Variable)`` returns False."""
    pass


class GradientTape:
    """Placeholder so ``isinstance(x, tf.GradientTape)`` returns False."""
    pass


# Common dtype constants
float16 = DType("float16")
float32 = DType("float32")
float64 = DType("float64")
bfloat16 = DType("bfloat16")
int8 = DType("int8")
int16 = DType("int16")
int32 = DType("int32")
int64 = DType("int64")
uint8 = DType("uint8")
uint16 = DType("uint16")
uint32 = DType("uint32")
uint64 = DType("uint64")
bool = DType("bool")
string = DType("string")
newaxis = None  # tf.newaxis is just None


def as_dtype(type_value):
    """Minimal stand-in for ``tf.as_dtype``."""
    if isinstance(type_value, DType):
        return type_value
    return DType(str(type_value))


def constant(value, dtype=None, shape=None, name=None):
    """Minimal stand-in for ``tf.constant`` — returns a numpy array."""
    import numpy as _np
    arr = _np.array(value)
    if dtype is not None:
        arr = arr.astype(str(dtype))
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def convert_to_tensor(value, dtype=None, name=None):
    """Minimal stand-in for ``tf.convert_to_tensor``."""
    import numpy as _np
    arr = _np.asarray(value)
    if dtype is not None:
        arr = arr.astype(str(dtype))
    return arr


def function(func=None, **kwargs):
    """No-op ``tf.function`` decorator — just returns the function."""
    if func is not None:
        return func
    def decorator(f):
        return f
    return decorator


class _NameScope:
    """Minimal stand-in for ``tf.name_scope``."""
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        return self.name
    def __exit__(self, *args):
        pass

name_scope = _NameScope


# ── Sub-namespace stubs ───────────────────────────────────────────────────────

class _DTypes:
    """Minimal ``tf.dtypes`` namespace."""
    float16 = DType("float16")
    float32 = DType("float32")
    float64 = DType("float64")
    bfloat16 = DType("bfloat16")
    int8 = DType("int8")
    int16 = DType("int16")
    int32 = DType("int32")
    int64 = DType("int64")
    uint8 = DType("uint8")
    bool = DType("bool")
    string = DType("string")
    DType = DType
    as_dtype = staticmethod(as_dtype)

dtypes = _DTypes()


class _Nest:
    """Minimal ``tf.nest`` namespace."""
    @staticmethod
    def flatten(structure):
        if isinstance(structure, dict):
            return list(structure.values())
        if isinstance(structure, (list, tuple)):
            return list(structure)
        return [structure]

    @staticmethod
    def pack_sequence_as(structure, flat_sequence):
        if isinstance(structure, dict):
            return dict(zip(structure.keys(), flat_sequence))
        if isinstance(structure, tuple):
            return tuple(flat_sequence)
        return list(flat_sequence)

    @staticmethod
    def map_structure(func, *structures):
        flat = [_Nest.flatten(s) for s in structures]
        return [func(*elems) for elems in zip(*flat)]

nest = _Nest()


class _Compat:
    """Minimal ``tf.compat`` namespace."""
    class v1:
        @staticmethod
        def reset_default_graph():
            pass

    class v2:
        pass

compat = _Compat()


class _Internal:
    """Minimal ``tf.__internal__`` namespace."""
    class tracking:
        @staticmethod
        def no_automatic_dependency_tracking(fn):
            return fn

__internal__ = _Internal()


class _Errors:
    """Minimal ``tf.errors`` namespace."""
    class OpError(Exception):
        pass
    class InvalidArgumentError(OpError):
        pass
    class NotFoundError(OpError):
        pass

errors = _Errors()


# ── 7. Expose top-level `tensorflow` attributes ───────────────────────────────
__version__ = "2.12.0-shim+keras3+torch"
__name__ = "tensorflow"

# Make `from tensorflow.keras import layers` etc. work by registering
# tensorflow.keras.* as proper submodules in sys.modules.
def _register_submodules():
    base = "tensorflow"
    # Register tensorflow itself
    sys.modules.setdefault(base, sys.modules[__name__])

    # Walk keras submodules and mirror them as tensorflow.keras.*
    _sub = [
        "keras",
        "keras.layers",
        "keras.models",
        "keras.optimizers",
        "keras.metrics",
        "keras.callbacks",
        "keras.regularizers",
        "keras.backend",
        "keras.utils",
        "keras.saving",
    ]
    for name in _sub:
        tf_name = f"tensorflow.{name}"
        keras_name = name  # e.g. "keras.layers"
        if tf_name not in sys.modules:
            # Try to get the real keras submodule
            try:
                parts = keras_name.split(".")
                mod = _keras
                for part in parts[1:]:  # skip 'keras' prefix
                    mod = getattr(mod, part)
                sys.modules[tf_name] = mod
            except AttributeError:
                # If not found, create an empty namespace to avoid ImportError
                sys.modules[tf_name] = types.ModuleType(tf_name)


_register_submodules()

# Also register our non-keras stub namespaces
_this = sys.modules[__name__]
for _attr_name in ("dtypes", "errors", "nest", "compat", "__internal__"):
    _tf_sub = f"tensorflow.{_attr_name}"
    if _tf_sub not in sys.modules:
        sys.modules[_tf_sub] = getattr(_this, _attr_name)
