"""Identity token verification for Sign in with Apple / Google OAuth.

``real`` verifiers here are genuine implementations (fetch the provider's
JWKS, verify the RS256 signature, check iss/aud/exp) -- but they are
untested beyond that: verifying them end to end needs a real Apple
Services ID / Google OAuth client and a real identity token minted by
Apple/Google's SDKs, which don't exist without the iOS app (v0.4) and a
registered app on each platform. Treat them as reviewed-but-unverified
until that exists, the same honesty bar as llm_serving/ before its GPU run.

``FakeIdentityVerifier`` is the default (AUTH_PROVIDER=fake) so the rest of
the auth flow -- token issuance, refresh, route protection -- is fully
testable without any of that. It accepts a JSON blob
'{"sub": "...", "email": "..."}' as the "identity token" and trusts it
outright; it must never be selectable in a deployment that has real users
(see AuthSettings' docstring on why nothing defaults to "real").
"""

import json
from dataclasses import dataclass
from typing import Protocol

import jwt
from jwt import PyJWKClient

from app.auth.config import auth_settings

APPLE_JWKS_URL = "https://appleid.apple.com/auth/keys"
APPLE_ISSUER = "https://appleid.apple.com"
GOOGLE_JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_ISSUERS = ("accounts.google.com", "https://accounts.google.com")


class IdentityVerificationError(Exception):
    pass


@dataclass
class VerifiedIdentity:
    provider_user_id: str
    email: str | None


class IdentityVerifier(Protocol):
    def verify(self, identity_token: str) -> VerifiedIdentity: ...


class AppleIdentityVerifier:
    """Verifies a Sign in with Apple identity token (a JWT signed by Apple)."""

    def __init__(self):
        self._jwks_client = PyJWKClient(APPLE_JWKS_URL)

    def verify(self, identity_token: str) -> VerifiedIdentity:
        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(identity_token)
            payload = jwt.decode(
                identity_token,
                signing_key.key,
                algorithms=["RS256"],
                audience=auth_settings.apple_client_id,
                issuer=APPLE_ISSUER,
            )
        except jwt.PyJWTError as exc:
            raise IdentityVerificationError(f"Apple identity token rejected: {exc}") from exc

        sub = payload.get("sub")
        if not sub:
            raise IdentityVerificationError("Apple identity token missing sub")
        return VerifiedIdentity(provider_user_id=sub, email=payload.get("email"))


class GoogleIdentityVerifier:
    """Verifies a Google Sign-In ID token (a JWT signed by Google)."""

    def __init__(self):
        self._jwks_client = PyJWKClient(GOOGLE_JWKS_URL)

    def verify(self, identity_token: str) -> VerifiedIdentity:
        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(identity_token)
            payload = jwt.decode(
                identity_token,
                signing_key.key,
                algorithms=["RS256"],
                audience=auth_settings.google_client_id,
            )
        except jwt.PyJWTError as exc:
            raise IdentityVerificationError(f"Google identity token rejected: {exc}") from exc

        if payload.get("iss") not in GOOGLE_ISSUERS:
            raise IdentityVerificationError(f"unexpected issuer: {payload.get('iss')!r}")

        sub = payload.get("sub")
        if not sub:
            raise IdentityVerificationError("Google identity token missing sub")
        return VerifiedIdentity(provider_user_id=sub, email=payload.get("email"))


class FakeIdentityVerifier:
    """Dev/test stand-in: trusts a JSON blob outright instead of verifying a
    real provider signature. See module docstring.
    """

    def verify(self, identity_token: str) -> VerifiedIdentity:
        try:
            data = json.loads(identity_token)
            sub = data["sub"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise IdentityVerificationError(
                'fake provider expects identity_token as JSON: {"sub": "...", "email": "..."}'
            ) from exc
        return VerifiedIdentity(provider_user_id=str(sub), email=data.get("email"))


def get_apple_verifier() -> IdentityVerifier:
    if auth_settings.provider == "fake":
        return FakeIdentityVerifier()
    return AppleIdentityVerifier()


def get_google_verifier() -> IdentityVerifier:
    if auth_settings.provider == "fake":
        return FakeIdentityVerifier()
    return GoogleIdentityVerifier()
