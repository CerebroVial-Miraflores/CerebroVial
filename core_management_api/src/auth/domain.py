"""Auth domain primitives.

Canonical role set per DHU-022: operator | manager | admin. The Role enum is the
single source of truth in the application layer; the DB column users.role stays
typed as String (no SQL Enum / CHECK) on purpose — validation happens here.
"""
from enum import Enum


class Role(str, Enum):
    OPERATOR = "operator"
    MANAGER = "manager"
    ADMIN = "admin"
