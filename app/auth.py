"""Shared API-key comparison; callers decide when to read request bodies."""
import hmac


def key_matches(presented: bytes, expected: bytes) -> bool:
    return bool(expected) and hmac.compare_digest(presented, expected)
