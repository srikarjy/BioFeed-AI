# TODO — path to interview-ready

Engineering is already resume-ready (v0.9, per BLUEPRINT.md §6): personalized
recsys + KG + explainability, 77 backend tests passing. What's missing is
what a recruiter/interviewer actually sees in the first 60 seconds, and the
job-search motion itself. Ordered by leverage, not by version number.

## 1. Make it look alive to a stranger (highest leverage, low effort)
- [ ] Record a 60–90s demo GIF/video against the **live** `/demo` page:
      search → browse/related → KG entities → sign-in → feed → `/explain`.
      Highest-ROI remaining item — most reviewers read a README for 30
      seconds and never clone the repo.
- [ ] Add 3–5 screenshots of the live `/demo` page to README under "What
      this is" — currently still prose-only with no visual proof.
- [x] ~~Add a "Try it in 60 seconds" block~~ — done: README now has curl
      commands against both the live URL and local `docker compose`.

## 2. Deploy something reachable by a URL
- [x] ~~Stand up a free-tier deployment~~ — done: `render.yaml` (Render
      Blueprint, free web service + free Postgres/pgvector), deployed from
      the existing GHCR image `.github/workflows/docker.yml` already
      publishes. `EMBEDDING_BACKEND=hash` and `LLM_ENABLED=false` for
      free-tier RAM/cost, documented in README's "Live deployment" section.
- [x] ~~Link the live URL at the top of README~~ — done.
- [ ] One-time account-side setup still needed (can't be done blind from the
      repo — see README's "Live deployment" → "One-time deploy setup"):
      create a GitHub PAT (`read:packages`), add it as a Render Registry
      Credential named `biofeed-ai-ghcr` (keeps the GHCR package private,
      per user preference), deploy the Blueprint from `render.yaml`, and
      store the resulting deploy-hook URL as the `RENDER_DEPLOY_HOOK_URL`
      GitHub secret so CI's new step actually fires.
- [ ] After first deploy: `curl -X POST .../ingest/run` once so the demo has
      real articles immediately instead of waiting for the first scheduled
      interval.

## 3. Close the one dangling "built but not surfaced" gap
- [x] ~~No UI consumes `/explain` yet~~ — done: `GET /demo`
      (`backend/app/static/demo.html`) is a same-origin static page that
      signs in via the fake identity provider and calls `/search`,
      `/articles`, `/kg/entities`, `/users/{id}/feed`, and
      `/users/{id}/articles/{id}/explain` live, with results rendered as
      cards. Turns "tested endpoint" into "thing you can click."

## 4. Tighten the pitch artifacts
- [ ] Rewrite resume bullets from BLUEPRINT.md §7 to lead with the
      generalist framing ("personalized content-ranking system with
      domain-specific embeddings," not "biotech platform") — the framing
      note already says this, just needs to land in the actual resume file.
- [ ] Write a 1-paragraph "elevator pitch" version for LinkedIn headline/
      About section and cover letters, distinct from the README's technical
      framing.
- [ ] Pick 2–3 "if they ask about X" talking points per hard decision in
      PROJECT_STATUS.md §3 (e.g. #22/#23 lazy-import + OpenMP segfault,
      #33 SQLite thread-safety) — these are the stories that differentiate
      you in a live interview, more than the feature list does.

## 5. Job-search motion (do in parallel, not after #1–4 are perfect)
- [ ] Finalize target list: recsys/ML-platform roles at companies where
      "content ranking + embeddings + explainability" is legibly relevant
      (not biotech-specific roles unless you want those).
- [ ] Get the repo + live URL + resume bullet into your LinkedIn "Featured"
      and into every application, not just ones you think will read it.
- [ ] Ask 1–2 people in ML/recsys roles for a 15-min review of the README
      + live demo before mass-applying — catches "why did you do X" gaps
      you can't see yourself.

## Not on this list (intentionally deferred)
- Mobile app (v0.4) and v2.0 market-signal module — both explicitly gated
  on real spend/Apple Developer account per BLUEPRINT.md, not worth
  blocking interview prep on.
- v1.0 full AWS production deployment — a free-tier deploy (#2) gets 90%
  of the interview value at 10% of the cost/effort.
