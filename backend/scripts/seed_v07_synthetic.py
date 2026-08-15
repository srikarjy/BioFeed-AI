"""Seed a synthetic corpus + synthetic users/interactions to exercise the
v0.7 training pipeline (train_v07.py) end to end.

There are zero real users (auth/mobile app haven't shipped yet — see
PROJECT_STATUS.md), so train_v07.py has nothing to train on against the real
database. This script generates a topic-clustered synthetic corpus and
synthetic users with topic-affinity-driven interactions, so the two-tower +
reranker training code path can actually run and be verified end to end
instead of staying an untested code path. It is explicitly NOT a substitute
for training on real interaction data — see the "not trained on real usage"
caveat in METRICS.md.

Usage:
    cd backend
    DATABASE_URL=sqlite:///./v07_seed.db EMBEDDING_BACKEND=hash \
        python scripts/seed_v07_synthetic.py
    DATABASE_URL=sqlite:///./v07_seed.db \
        python scripts/train_v07.py --output-dir models/v0.7-synthetic --min-interactions 5
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import Base, SessionLocal, engine
from app.models import Article
from app.ml.embeddings import get_embedder
from app import crud
from app.schemas import ArticleCreate, InteractionType

random.seed(42)

# Topic clusters: (source, subject noun phrases, verb phrases) so titles
# stay on-topic but non-repetitive. Distinct vocabulary per topic matters
# more than realism here -- it's what gives the trained models a signal to
# learn from.
TOPICS: dict[str, dict] = {
    "gene_therapy": {
        "subjects": ["CRISPR gene editing", "AAV gene therapy", "base editing", "in vivo lentiviral therapy"],
        "conditions": ["sickle cell disease", "beta-thalassemia", "Duchenne muscular dystrophy", "inherited blindness", "hemophilia"],
        "sources": ["STAT News", "GEN", "FiercePharma"],
    },
    "immuno_oncology": {
        "subjects": ["a PD-1 checkpoint inhibitor", "a CAR-T cell therapy", "a bispecific antibody", "an antibody-drug conjugate"],
        "conditions": ["metastatic melanoma", "relapsed lymphoma", "solid tumors", "advanced breast cancer", "multiple myeloma"],
        "sources": ["STAT News", "BioPharma Dive", "GEN"],
    },
    "metabolic_disease": {
        "subjects": ["a GLP-1 receptor agonist", "an oral GLP-1 pill", "a dual GIP/GLP-1 agonist", "an amylin analog"],
        "conditions": ["obesity", "type 2 diabetes", "cardiovascular risk", "fatty liver disease"],
        "sources": ["STAT News", "FiercePharma"],
    },
    "mrna_platform": {
        "subjects": ["an mRNA vaccine candidate", "a next-gen lipid nanoparticle", "a self-amplifying mRNA platform"],
        "conditions": ["seasonal flu", "RSV", "non-hepatic drug delivery", "personalized cancer vaccines"],
        "sources": ["GEN", "BioPharma Dive"],
    },
    "ai_drug_discovery": {
        "subjects": ["a generative-AI drug discovery platform", "a machine learning toxicity model", "an AI-designed small molecule", "a protein-structure prediction model"],
        "conditions": ["fibrosis", "drug-induced liver injury", "novel kinase targets", "antibody design"],
        "sources": ["GEN", "BioPharma Dive"],
    },
    "biosimilars_pricing": {
        "subjects": ["a new biosimilar", "an interchangeable biosimilar", "a follow-on biologic"],
        "conditions": ["autoimmune disease", "rheumatoid arthritis", "diabetes", "oncology supportive care"],
        "sources": ["FiercePharma", "STAT News"],
    },
    "microbiome": {
        "subjects": ["a live biotherapeutic product", "a microbiome-based therapy", "a fecal microbiota transplant"],
        "conditions": ["recurrent C. difficile infection", "immunotherapy response", "inflammatory bowel disease"],
        "sources": ["GEN", "BioPharma Dive"],
    },
    "neurodegeneration": {
        "subjects": ["an anti-amyloid antibody", "a tau-targeting therapy", "an ASO for neurodegeneration"],
        "conditions": ["early Alzheimer's disease", "ALS", "Parkinson's disease"],
        "sources": ["STAT News", "FiercePharma"],
    },
    "biotech_business": {
        "subjects": ["a $4 billion acquisition", "a Series C funding round", "an IPO filing", "a licensing deal"],
        "conditions": ["oncology pipeline expansion", "the funding environment", "biotech valuations", "manufacturing capacity"],
        "sources": ["BioPharma Dive", "FiercePharma", "STAT News"],
    },
    "infectious_disease": {
        "subjects": ["a novel antibiotic candidate", "a broad-spectrum antiviral", "an updated vaccine"],
        "conditions": ["drug-resistant bacteria", "a new viral variant", "hospital-acquired infection"],
        "sources": ["STAT News", "GEN"],
    },
}

STAGES = ["preclinical study", "phase 1 trial", "phase 2 trial", "phase 3 trial", "FDA review"]


def generate_corpus() -> list[dict]:
    articles = []
    article_id = 0
    for topic, spec in TOPICS.items():
        for subject in spec["subjects"]:
            for condition in spec["conditions"]:
                stage = STAGES[article_id % len(STAGES)]
                source = spec["sources"][article_id % len(spec["sources"])]
                title = f"{subject.capitalize()} advances to {stage} for {condition}"
                summary = f"New data on {subject} in {condition} were reported from a {stage}, adding to the {topic.replace('_', ' ')} pipeline."
                articles.append({"topic": topic, "title": title, "summary": summary, "source": source})
                article_id += 1
    return articles


def seed_corpus(db) -> dict[str, list[int]]:
    """Insert articles, embed them, return topic -> [article_id] map."""
    corpus = generate_corpus()
    embedder = get_embedder()
    texts = [f"{a['title']} {a['summary']}" for a in corpus]
    vectors = embedder.embed_texts(texts)

    by_topic: dict[str, list[int]] = {t: [] for t in TOPICS}
    for row, vector in zip(corpus, vectors):
        article, _ = crud.create_article(
            db,
            ArticleCreate(
                title=row["title"],
                url=f"https://seed.local/{row['topic']}/{row['title']}",
                source=row["source"],
                summary=row["summary"],
            ),
        )
        article.embedding = vector
        db.commit()
        by_topic[row["topic"]].append(article.id)

    print(f"Seeded {len(corpus)} synthetic articles across {len(TOPICS)} topics")
    return by_topic


def seed_users(db, by_topic: dict[str, list[int]], num_users: int = 25):
    topics = list(by_topic.keys())
    for i in range(num_users):
        primary = topics[i % len(topics)]
        secondary = topics[(i + 3) % len(topics)]
        user = crud.create_user(db, email=f"synthetic-user-{i}@eval.local")

        primary_ids = by_topic[primary]
        secondary_ids = by_topic[secondary]

        # Primary-topic interactions: the bulk of the signal.
        for aid in random.sample(primary_ids, k=min(10, len(primary_ids))):
            action = random.choices(
                [InteractionType.READ, InteractionType.BOOKMARK, InteractionType.LIKE],
                weights=[0.6, 0.25, 0.15],
            )[0]
            read_time = random.randint(60, 300) if action == InteractionType.READ else None
            crud.create_interaction(db, user.id, aid, action, read_time_seconds=read_time)

        # Secondary-topic interactions: weaker, occasional signal.
        for aid in random.sample(secondary_ids, k=min(4, len(secondary_ids))):
            crud.create_interaction(db, user.id, aid, InteractionType.READ, read_time_seconds=random.randint(30, 120))

        # A couple of hides, off-topic, to exercise the exclusion path.
        off_topic = topics[(i + 5) % len(topics)]
        for aid in random.sample(by_topic[off_topic], k=min(2, len(by_topic[off_topic]))):
            crud.create_interaction(db, user.id, aid, InteractionType.HIDE)

        crud.refresh_user_embedding(db, user.id)

    print(f"Seeded {num_users} synthetic users with topic-affinity interactions")


def main():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        by_topic = seed_corpus(db)
        seed_users(db, by_topic)
    finally:
        db.close()


if __name__ == "__main__":
    main()
