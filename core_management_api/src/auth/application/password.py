"""Password hashing using bcrypt (CT-01.2).

bcrypt cost factor is fixed at 12 (>= the spec's minimum). We call the bcrypt
package directly instead of going through passlib because passlib 1.7.4 has a
startup-time incompatibility with bcrypt >= 5 (it tries to hash a 120-byte
payload during backend detection, which bcrypt 5.x now rejects).
"""
import bcrypt

BCRYPT_COST = 12


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(
        plain.encode("utf-8"),
        bcrypt.gensalt(rounds=BCRYPT_COST),
    ).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False
