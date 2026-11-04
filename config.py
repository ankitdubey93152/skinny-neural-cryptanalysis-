"""Central configuration for SKINNY-128-128 neural cryptanalysis."""
from typing import List

BLOCK_BYTES: int = 16
BLOCK_BITS: int = 128
KEY_BYTES: int = 16
PAIR_BITS: int = 256

DEFAULT_ROUNDS: int = 40
DEFAULT_ROUND_SWEEP: List[int] = [1, 2, 3, 4, 5, 6, 8, 10, 16, 20, 32, 40]
DEFAULT_DELTA: bytes = bytes.fromhex("00000000000000000000000000000001")
DEFAULT_SEED: int = 42
