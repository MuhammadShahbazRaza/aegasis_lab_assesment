import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("DATABASE_URL", "sqlite:///./search_intel.db")

from app.config import get_settings  # noqa: E402
from app.db.base import init_db, session_scope  # noqa: E402
from app.db.repository import (  # noqa: E402
    ProfileRepository,
    QueryRepository,
    RecommendationRepository,
)
from app.observability.logging import configure_logging  # noqa: E402
from app.service import PipelineService  # noqa: E402

PROFILE = {
    "name": "Surfer SEO",
    "domain": "surferseo.com",
    "industry": "SEO Software",
    "description": "AI-powered SEO content optimization tool",
    "competitors": ["clearscope.io", "marketmuse.com", "frase.io"],
}


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    init_db()

    with session_scope() as session:
        repo = ProfileRepository(session)
        profile = repo.find_by_domain(PROFILE["domain"]) or repo.create(**PROFILE)
        session.flush()
        run = PipelineService(session, settings).run_profile(profile)
        session.flush()

        rows, total = QueryRepository(session).list_for_run(run.run_uuid, per_page=10)
        recs = RecommendationRepository(session).list_for_run(run.run_uuid)

        print("\n" + "=" * 78)
        print(f"run {run.run_uuid}  status={run.status}  degraded={run.degraded}")
        print(f"path: {' -> '.join(run.node_path or [])}")
        print(
            f"planned={run.planned_call_count} normalized={run.normalized_record_count} "
            f"queries={total} recommendations={len(recs)} "
            f"tokens={run.token_usage} duration={run.duration_ms:.0f}ms"
        )
        print("=" * 78)
        for row in rows[:8]:
            print(
                f"  {row.opportunity_score:.3f}  vol={row.estimated_search_volume:>7,}  "
                f"diff={row.competitive_difficulty:>3}  {row.visibility_status:<12} "
                f"{row.query_text}"
            )
        print("-" * 78)
        for rec in recs[:5]:
            print(f"  [{rec.priority:<6}] {rec.content_type:<17} {rec.title}")
        print("-" * 78)
        print(json.dumps(run.metrics, indent=2)[:1400])
        if run.report:
            print("\n" + str(run.report.get("summary_markdown", ""))[:1200])
    return 0 if run.status != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
