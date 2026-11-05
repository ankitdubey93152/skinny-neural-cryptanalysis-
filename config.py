"""Central configuration for SKINNY-128-128 neural cryptanalysis and multi-cipher extensions."""
from typing import List, Dict, Any

BLOCK_BYTES: int = 16
BLOCK_BITS: int = 128
KEY_BYTES: int = 16
PAIR_BITS: int = 256

DEFAULT_ROUNDS: int = 40
DEFAULT_ROUND_SWEEP: List[int] = [1, 2, 3, 4, 5, 6, 8, 10, 16, 20, 32, 40]
DEFAULT_DELTA: bytes = bytes.fromhex("00000000000000000000000000000001")
DEFAULT_SEED: int = 42

# Extended small-sample experiment configurations (D, E, F)
SAMPLE_SIZES_SMALL: List[int] = [50, 500, 5000]
SAMPLE_SIZE_D: int = 50
SAMPLE_SIZE_E: int = 500
SAMPLE_SIZE_F: int = 5000
DEFAULT_REPLICATES: Dict[int, int] = {50: 30, 500: 20, 5000: 10}

# Extended multi-cipher experiment configurations (G)
DEFAULT_DELTA_64: bytes = bytes.fromhex("0000000000000001")
CIPHERS_NATIVE_128: List[str] = ["skinny128", "aes128", "sm4"]
CIPHERS_COMMON_64: List[str] = ["skinny128", "aes128", "sm4", "hight", "des"]

CIPHER_METADATA: Dict[str, Dict[str, Any]] = {
    "skinny128": {"block_bytes": 16, "key_bytes": 16, "full_rounds": 40, "name": "SKINNY-128-128"},
    "aes128": {"block_bytes": 16, "key_bytes": 16, "full_rounds": 10, "name": "AES-128"},
    "sm4": {"block_bytes": 16, "key_bytes": 16, "full_rounds": 32, "name": "SM4 (GB/T 32907-2016)"},
    "hight": {"block_bytes": 8, "key_bytes": 16, "full_rounds": 32, "name": "HIGHT (ISO/IEC 18033-3)"},
    "des": {"block_bytes": 8, "key_bytes": 8, "full_rounds": 16, "name": "DES (FIPS 46-3)"},
}
