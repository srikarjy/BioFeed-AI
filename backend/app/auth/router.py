from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app import crud
from app.auth.dependencies import get_current_user
from app.auth.jwt import TokenError, create_access_token, create_refresh_token, decode_token
from app.auth.providers import IdentityVerificationError, get_apple_verifier, get_google_verifier
from app.database import get_db
from app.models import User
from app.schemas import UserRead

router = APIRouter(prefix="/auth", tags=["auth"])


class IdentityTokenRequest(BaseModel):
    identity_token: str
    display_name: str | None = None


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserRead


class RefreshRequest(BaseModel):
    refresh_token: str


class AccessTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


def _issue_tokens(user: User) -> TokenPair:
    return TokenPair(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
        user=UserRead.model_validate(user),
    )


@router.post("/apple", response_model=TokenPair)
def sign_in_with_apple(body: IdentityTokenRequest, db: Session = Depends(get_db)):
    try:
        identity = get_apple_verifier().verify(body.identity_token)
    except IdentityVerificationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    user = crud.get_or_create_user(
        db,
        email=identity.email,
        apple_user_id=identity.provider_user_id,
        display_name=body.display_name,
    )
    return _issue_tokens(user)


@router.post("/google", response_model=TokenPair)
def sign_in_with_google(body: IdentityTokenRequest, db: Session = Depends(get_db)):
    try:
        identity = get_google_verifier().verify(body.identity_token)
    except IdentityVerificationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    user = crud.get_or_create_user(
        db,
        email=identity.email,
        google_user_id=identity.provider_user_id,
        display_name=body.display_name,
    )
    return _issue_tokens(user)


@router.post("/refresh", response_model=AccessTokenResponse)
def refresh_access_token(body: RefreshRequest, db: Session = Depends(get_db)):
    try:
        user_id = decode_token(body.refresh_token, expected_type="refresh")
    except TokenError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid refresh token: {exc}") from exc

    user = crud.get_user(db, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    return AccessTokenResponse(access_token=create_access_token(user.id))


@router.get("/me", response_model=UserRead)
def read_current_user(current_user: User = Depends(get_current_user)):
    return current_user
