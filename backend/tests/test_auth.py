"""Tests for v0.3 auth: token issuance/refresh, and enforcement on the
user-scoped v0.6/v0.7 routes.

Runs against the fake identity provider (AUTH_PROVIDER=fake, the default),
which trusts a JSON blob as the "identity token" instead of verifying a
real Apple/Google signature -- see app/auth/providers.py for why the real
verifiers can't be exercised without a registered app and a real device.
"""

import json

import pytest

from app.auth.jwt import TokenError, create_access_token, create_refresh_token, decode_token


def _fake_identity_token(sub: str, email: str | None = None) -> str:
    return json.dumps({"sub": sub, "email": email})


def test_sign_in_with_apple_creates_user_and_issues_tokens(client):
    resp = client.post(
        "/auth/apple",
        json={"identity_token": _fake_identity_token("apple-sub-1", "a@example.com")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["refresh_token"]
    assert body["user"]["email"] == "a@example.com"


def test_sign_in_with_apple_is_idempotent_for_same_identity(client):
    r1 = client.post("/auth/apple", json={"identity_token": _fake_identity_token("apple-sub-2")})
    r2 = client.post("/auth/apple", json={"identity_token": _fake_identity_token("apple-sub-2")})
    assert r1.json()["user"]["id"] == r2.json()["user"]["id"]


def test_sign_in_with_google_creates_user(client):
    resp = client.post(
        "/auth/google",
        json={"identity_token": _fake_identity_token("google-sub-1", "g@example.com")},
    )
    assert resp.status_code == 200
    assert resp.json()["user"]["email"] == "g@example.com"


def test_malformed_identity_token_rejected(client):
    resp = client.post("/auth/apple", json={"identity_token": "not-json"})
    assert resp.status_code == 401


def test_me_requires_bearer_token(client):
    assert client.get("/auth/me").status_code == 401
    assert client.get("/auth/me", headers={"Authorization": "garbage"}).status_code == 401


def test_me_returns_current_user(client):
    signup = client.post("/auth/apple", json={"identity_token": _fake_identity_token("apple-sub-3")})
    token = signup.json()["access_token"]
    resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["id"] == signup.json()["user"]["id"]


def test_refresh_issues_new_access_token(client):
    signup = client.post("/auth/apple", json={"identity_token": _fake_identity_token("apple-sub-4")})
    refresh_token = signup.json()["refresh_token"]

    resp = client.post("/auth/refresh", json={"refresh_token": refresh_token})
    assert resp.status_code == 200
    new_access = resp.json()["access_token"]

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {new_access}"})
    assert me.status_code == 200


def test_access_token_cannot_be_used_as_refresh_token(client):
    signup = client.post("/auth/apple", json={"identity_token": _fake_identity_token("apple-sub-5")})
    access_token = signup.json()["access_token"]

    resp = client.post("/auth/refresh", json={"refresh_token": access_token})
    assert resp.status_code == 401


def test_refresh_token_cannot_be_used_as_access_token(client):
    signup = client.post("/auth/apple", json={"identity_token": _fake_identity_token("apple-sub-6")})
    refresh_token = signup.json()["refresh_token"]

    resp = client.get("/auth/me", headers={"Authorization": f"Bearer {refresh_token}"})
    assert resp.status_code == 401


def test_decode_token_rejects_wrong_type():
    token = create_access_token(1)
    with pytest.raises(TokenError):
        decode_token(token, expected_type="refresh")


def test_decode_token_rejects_garbage():
    with pytest.raises(TokenError):
        decode_token("not-a-jwt", expected_type="access")


def test_feed_requires_authentication(client):
    signup = client.post("/auth/apple", json={"identity_token": _fake_identity_token("apple-sub-7")})
    user_id = signup.json()["user"]["id"]

    resp = client.get(f"/users/{user_id}/feed")
    assert resp.status_code == 401


def test_feed_rejects_mismatched_user(client):
    me = client.post("/auth/apple", json={"identity_token": _fake_identity_token("apple-sub-8")})
    other = client.post("/auth/apple", json={"identity_token": _fake_identity_token("apple-sub-9")})

    my_token = me.json()["access_token"]
    other_user_id = other.json()["user"]["id"]

    resp = client.get(f"/users/{other_user_id}/feed", headers={"Authorization": f"Bearer {my_token}"})
    assert resp.status_code == 403


def test_feed_allows_authenticated_self(client):
    signup = client.post("/auth/apple", json={"identity_token": _fake_identity_token("apple-sub-10")})
    user_id = signup.json()["user"]["id"]
    token = signup.json()["access_token"]

    resp = client.get(f"/users/{user_id}/feed", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200


def test_interactions_require_self(client, db_session):
    from app.schemas import ArticleCreate
    from app import crud

    signup = client.post("/auth/apple", json={"identity_token": _fake_identity_token("apple-sub-11")})
    user_id = signup.json()["user"]["id"]
    token = signup.json()["access_token"]

    article, _ = crud.create_article(
        db_session,
        ArticleCreate(title="Test article", url="http://x/test-auth-article", source="test"),
    )

    # No auth at all.
    resp = client.post(
        f"/users/{user_id}/interactions",
        json={"article_id": article.id, "interaction_type": "like"},
    )
    assert resp.status_code == 401

    # Authenticated as self.
    resp = client.post(
        f"/users/{user_id}/interactions",
        json={"article_id": article.id, "interaction_type": "like"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
