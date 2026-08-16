from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app import crud
from app.auth.jwt import TokenError, decode_token
from app.database import get_db
from app.models import User


def get_current_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    """Resolve the caller's User from a `Bearer <access token>` header.

    401 for anything wrong with the token (missing, malformed, wrong type,
    expired, unknown signature) or a user id that no longer exists -- a
    revoked/deleted user shouldn't be distinguishable from a bad token.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = authorization[len("Bearer ") :]
    try:
        user_id = decode_token(token, expected_type="access")
    except TokenError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}") from exc

    user = crud.get_user(db, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user


def require_self(user_id: int, current_user: User = Depends(get_current_user)) -> User:
    """Route dependency for `/users/{user_id}/...` endpoints: the
    authenticated caller must BE that user. Returns the user so route
    handlers that already do `crud.get_user(db, user_id)` don't need a
    second lookup.
    """
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized for this user")
    return current_user
