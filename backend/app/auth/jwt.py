"""Access/refresh token issuance and verification.

Two token types share one HS256-signed format (``jwt_secret`` from
AuthSettings), distinguished by a ``type`` claim so a refresh token can't be
replayed as an access token or vice versa.
"""

from datetime import datetime, timedelta, timezone
from typing import Literal

import jwt

from app.auth.config import auth_settings

TokenType = Literal["access", "refresh"]


class TokenError(Exception):
    pass


def _create_token(user_id: int, token_type: TokenType, ttl: timedelta) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "type": token_type,
        "iat": now,
        "exp": now + ttl,
    }
    return jwt.encode(payload, auth_settings.jwt_secret, algorithm=auth_settings.jwt_algorithm)


def create_access_token(user_id: int) -> str:
    return _create_token(user_id, "access", timedelta(minutes=auth_settings.access_token_ttl_minutes))


def create_refresh_token(user_id: int) -> str:
    return _create_token(user_id, "refresh", timedelta(days=auth_settings.refresh_token_ttl_days))


def decode_token(token: str, expected_type: TokenType) -> int:
    """Return the user id encoded in a valid, non-expired token of the
    expected type. Raises TokenError otherwise -- callers turn this into a
    401, not a 500.
    """
    try:
        payload = jwt.decode(token, auth_settings.jwt_secret, algorithms=[auth_settings.jwt_algorithm])
    except jwt.PyJWTError as exc:
        raise TokenError(str(exc)) from exc

    if payload.get("type") != expected_type:
        raise TokenError(f"expected a {expected_type} token, got {payload.get('type')!r}")

    try:
        return int(payload["sub"])
    except (KeyError, ValueError) as exc:
        raise TokenError("token missing a valid subject") from exc
