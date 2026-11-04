"""Global random seeding and hardware device management utilities."""
import os, random, numpy as np
from typing import Dict, Any

def detect_hardware() -> Dict[str, Any]:
    """Detect available compute devices (GPU / CPU) and return device info."""
    info = {"device_type": "CPU", "gpus_available": [], "gpu_count": 0, "cuda_built": False}
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        info.update({"cuda_built": tf.test.is_built_with_cuda(), "gpu_count": len(gpus)})
        if gpus:
            info["device_type"] = "GPU"
            info["gpus_available"] = [g.name for g in gpus]
            for g in gpus:
                try: tf.config.experimental.set_memory_growth(g, True)
                except Exception: pass
    except Exception as e:
        info["error"] = str(e)
    return info

def set_global_seed(seed: int = 42, enable_op_determinism: bool = False) -> Dict[str, Any]:
    """Set global random seeds for Python, NumPy, and TensorFlow."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    np.random.seed(seed)
    status = {"seed": seed, "python_seeded": True, "numpy_seeded": True, "tf_seeded": False, "deterministic_ops_enabled": False}
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        status["tf_seeded"] = True
        if enable_op_determinism:
            try:
                tf.config.experimental.enable_op_determinism()
                status["deterministic_ops_enabled"] = True
            except Exception as e:
                status["op_determinism_warning"] = str(e)
    except ImportError:
        pass
    status["hardware"] = detect_hardware()
    return status
