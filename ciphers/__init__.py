"""Cipher implementations used by the neural cryptanalysis framework."""
from ciphers.skinny128 import skinny128_encrypt, skinny128_encrypt_batch
from ciphers.aes_ref import aes128_encrypt_batch
from ciphers.sm4 import sm4_encrypt, sm4_encrypt_batch
from ciphers.hight import hight_encrypt, hight_encrypt_batch
from ciphers.des import des_encrypt, des_encrypt_batch

__all__ = [
    "skinny128_encrypt",
    "skinny128_encrypt_batch",
    "aes128_encrypt_batch",
    "sm4_encrypt",
    "sm4_encrypt_batch",
    "hight_encrypt",
    "hight_encrypt_batch",
    "des_encrypt",
    "des_encrypt_batch",
]
