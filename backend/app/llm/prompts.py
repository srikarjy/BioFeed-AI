from app.anomaly.models import AnomalyEvent
from app.models import Article


def build_explanation_prompt(
    event: AnomalyEvent, article: Article, related: list[Article]
) -> str:
    """Ground the prompt in the detector's actual evidence, not just the title,
    so the model explains the specific signal rather than free-associating.
    """
    related_lines = "\n".join(f"- [{a.source}] {a.title}" for a in related) or "(none)"
    return (
        "You are a biotech news analyst. In 2-3 sentences, explain why the "
        "following article was flagged as a candidate early-signal event. "
        "Ground the explanation in the evidence given; do not invent facts "
        "not present below.\n\n"
        f"Flagged article ({article.source}): {article.title}\n"
        f"Summary: {article.summary or '(none)'}\n\n"
        f"Detector: {event.kind}\n"
        f"Score: {event.score}\n"
        f"Evidence: {len(event.detail.get('related_sources', []))} independent "
        f"source(s) published closely matching coverage within "
        f"{event.detail.get('window_hours', '?')} hours "
        f"(mean similarity {event.detail.get('mean_similarity', '?')}).\n\n"
        f"Corroborating coverage:\n{related_lines}\n\n"
        "Explanation:"
    )
