"""Cipher implementations used by the SKINNY neural cryptanalysis project."""
from ciphers.skinny128 import skinny128_encrypt, skinny128_encrypt_batch
from ciphers.aes_ref import aes128_encrypt_batch

__all__ = ["skinny128_encrypt", "skinny128_encrypt_batch", "aes128_encrypt_batch"]
