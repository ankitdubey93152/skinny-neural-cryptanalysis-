"""Deep learning neural distinguisher architectures (MLP, 1D CNN, 2D CNN)."""
from typing import Tuple, Union, Sequence, Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

DEFAULT_UNITS: int = 640
DEFAULT_FILTERS_1D: Tuple[int, ...] = (32, 64)
DEFAULT_FILTERS_2D: Tuple[int, ...] = (32, 64)
DEFAULT_KERNEL_SIZE_1D: int = 3
DEFAULT_KERNEL_SIZE_2D: int = 3
DEFAULT_LEARNING_RATE: float = 1e-3
DEFAULT_USE_BATCH_NORM: bool = True
DEFAULT_POOLING: str = "flatten"


def build_mlp(input_dim: Union[int, Tuple[int, ...]], n_dense_layers: int = 3, units: int = DEFAULT_UNITS, learning_rate: float = DEFAULT_LEARNING_RATE) -> tf.keras.Model:
    """Construct Multi-Layer Perceptron (MLP) distinguisher."""
    shape = (input_dim,) if isinstance(input_dim, int) else (input_dim if len(input_dim) == 1 else (int(np.prod(input_dim)),))
    inputs = layers.Input(shape=shape, name="mlp_input")
    x = inputs
    for i in range(n_dense_layers):
        x = layers.Dense(units, activation="relu", name=f"dense_{i + 1}")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output_sigmoid")(x)
    m = models.Model(inputs=inputs, outputs=outputs, name=f"MLP_L{n_dense_layers}_U{units}")
    m.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
    return m


def build_cnn_1d(input_shape: Tuple[int, int], n_dense_layers: int = 2, filters: Sequence[int] = DEFAULT_FILTERS_1D, kernel_size: int = DEFAULT_KERNEL_SIZE_1D, dense_units: int = DEFAULT_UNITS, use_batch_norm: bool = DEFAULT_USE_BATCH_NORM, pooling: str = DEFAULT_POOLING, learning_rate: float = DEFAULT_LEARNING_RATE) -> tf.keras.Model:
    """Construct 1D Convolutional Neural Network distinguisher."""
    inputs = layers.Input(shape=input_shape, name="cnn1d_input")
    x = inputs
    for i, nf in enumerate(filters):
        x = layers.Conv1D(filters=nf, kernel_size=kernel_size, padding="same", name=f"conv1d_{i + 1}")(x)
        if use_batch_norm:
            x = layers.BatchNormalization(name=f"bn_{i + 1}")(x)
        x = layers.ReLU(name=f"relu_{i + 1}")(x)
    x = layers.GlobalAveragePooling1D(name="gap1d")(x) if pooling == "gap" else layers.Flatten(name="flatten")(x)
    for i in range(n_dense_layers):
        x = layers.Dense(dense_units, activation="relu", name=f"dense_{i + 1}")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output_sigmoid")(x)
    m = models.Model(inputs=inputs, outputs=outputs, name=f"CNN1D_L{n_dense_layers}_F{'-'.join(map(str, filters))}")
    m.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
    return m


def build_cnn_2d(input_shape: Tuple[int, int, int], n_dense_layers: int = 2, kernel_size: int = DEFAULT_KERNEL_SIZE_2D, filters: Sequence[int] = DEFAULT_FILTERS_2D, dense_units: int = DEFAULT_UNITS, use_batch_norm: bool = DEFAULT_USE_BATCH_NORM, pooling: str = DEFAULT_POOLING, learning_rate: float = DEFAULT_LEARNING_RATE) -> tf.keras.Model:
    """Construct 2D Convolutional Neural Network distinguisher."""
    inputs = layers.Input(shape=input_shape, name="cnn2d_input")
    x = inputs
    for i, nf in enumerate(filters):
        x = layers.Conv2D(filters=nf, kernel_size=(kernel_size, kernel_size), padding="same", name=f"conv2d_{i + 1}")(x)
        if use_batch_norm:
            x = layers.BatchNormalization(name=f"bn_{i + 1}")(x)
        x = layers.ReLU(name=f"relu_{i + 1}")(x)
    x = layers.GlobalAveragePooling2D(name="gap2d")(x) if pooling == "gap" else layers.Flatten(name="flatten")(x)
    for i in range(n_dense_layers):
        x = layers.Dense(dense_units, activation="relu", name=f"dense_{i + 1}")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output_sigmoid")(x)
    m = models.Model(inputs=inputs, outputs=outputs, name=f"CNN2D_L{n_dense_layers}_K{kernel_size}_F{'-'.join(map(str, filters))}")
    m.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
    return m


def prepare_input(X_bits: np.ndarray, model_type: str, experiment: str, block_bits: int = 128) -> np.ndarray:
    """Format bit tensors to match the expected input shape for each model architecture."""
    m_type, exp = model_type.lower().strip(), experiment.lower().strip().replace("exp_", "")
    N, X_f, p_bits = X_bits.shape[0], X_bits.astype(np.float32), 2 * block_bits

    if exp in ("a", "c"):
        if m_type == "mlp":
            return X_f.reshape((N, p_bits))
        if m_type == "cnn_1d":
            return np.stack([X_f[:, :block_bits], X_f[:, block_bits:p_bits]], axis=-1)
        if m_type == "cnn_2d":
            return np.expand_dims(np.stack([X_f[:, :block_bits], X_f[:, block_bits:p_bits]], axis=1), axis=-1)

    elif exp == "b":
        n_pairs = X_bits.shape[1] if X_bits.ndim > 2 else 4
        if m_type == "mlp":
            return X_f.reshape((N, n_pairs * p_bits))
        if m_type == "cnn_1d":
            X_p = X_f.reshape((N, n_pairs, 2, block_bits))
            return np.transpose(X_p, (0, 3, 1, 2)).reshape((N, block_bits, n_pairs * 2))
        if m_type == "cnn_2d":
            return np.expand_dims(X_f.reshape((N, n_pairs, p_bits)), axis=-1)

    raise ValueError(f"Unknown combination of model_type='{model_type}' and experiment='{experiment}'")
