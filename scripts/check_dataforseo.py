"""Verify DataForSEO credentials and connectivity before running the pipeline.

Makes one cheap live call per selected endpoint and reports what came back, so a bad
password or a wrong base URL surfaces here rather than halfway through a DAG run.

    DATAFORSEO_MODE=live DATAFORSEO_BASE_URL=https://sandbox.dataforseo.com \
        .venv/bin/python scripts/check_dataforseo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.clients.dataforseo import LiveDataForSEOClient  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.observability.logging import configure_logging  # noqa: E402
from app.resilience.errors import ToolExecutionError  # noqa: E402
from app.tools.registry import TOOLS, validate_tool_call  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

# One representative, cheapest-form payload per tool.
PROBES: dict[str, dict] = {
    "keyword_search_volume": {
        "keywords": ["best seo software"],
        "location_name": "United States",
        "language_code": "en",
    },
    "serp_organic_results": {
        "keyword": "best seo software",
        "location_name": "United States",
        "language_code": "en",
        "depth": 10,
    },
    "ai_overview_snapshot": {
        "keyword": "best seo software",
        "location_name": "United States",
        "language_code": "en",
    },
    "llm_answer_visibility": {
        "user_prompt": "What is the best SEO software?",
        "model_name": "gpt-4o-mini",
        "web_search": True,
    },
    "related_keyword_ideas": {
        "seed_keywords": ["seo software"],
        "location_name": "United States",
        "language_code": "en",
        "limit": 5,
    },
}

HINTS = {
    401: "check DATAFORSEO_LOGIN / DATAFORSEO_PASSWORD - the API password from "
         "app.dataforseo.com/api-access is not your account password",
    402: "account balance is empty; top up or use the sandbox base URL",
    403: "the account exists but is not verified yet - finish verification at "
         "app.dataforseo.com; this blocks the sandbox too",
    404: "endpoint path not found - check DATAFORSEO_BASE_URL has no trailing /v3",
}


def main() -> int:
    settings = get_settings()
    configure_logging("WARNING", "console")

    if settings.dataforseo_mode != "live":
        print(f"{YELLOW}DATAFORSEO_MODE is '{settings.dataforseo_mode}'.{RESET} "
              "Set it to 'live' to test real credentials.")
        return 2
    if settings.dataforseo_credentials is None:
        print(f"{RED}missing credentials{RESET} - set DATAFORSEO_LOGIN and "
              "DATAFORSEO_PASSWORD in .env")
        return 2

    sandbox = "sandbox" in settings.dataforseo_base_url
    mode = (
        "sandbox: free, dummy values, real response envelopes"
        if sandbox
        else "PRODUCTION: these calls are billed"
    )
    print(f"→ {settings.dataforseo_base_url}  login={settings.dataforseo_login}")
    print(f"{DIM}{mode}{RESET}\n")

    selected = sys.argv[1:] or list(PROBES)
    client = LiveDataForSEOClient(settings)
    failures = 0
    total_cost = 0.0

    try:
        for name in selected:
            spec = TOOLS.get(name)
            if spec is None:
                print(f"  {RED}FAIL{RESET} unknown tool '{name}'")
                failures += 1
                continue
            try:
                # Go through the same validation and field-mapping the executor uses,
                # so this verifies the real request path rather than a hand-built one.
                call = validate_tool_call(name, PROBES[name])
                body = client.execute(
                    spec.endpoint, [call.spec.to_payload(call.payload)], tool=name
                )
            except ToolExecutionError as exc:
                status = exc.context.status_code or exc.context.provider_code
                hint = HINTS.get(int(str(status)[:3]) if status else 0, "")
                print(f"  {RED}FAIL{RESET} {name:<24} {exc.message}")
                if hint:
                    print(f"         {DIM}{hint}{RESET}")
                failures += 1
                continue

            task = (body.get("tasks") or [{}])[0]
            results = task.get("result") or []
            items = results[0].get("items") if results and isinstance(results[0], dict) else None
            count = len(items) if isinstance(items, list) else len(results)
            cost = float(body.get("cost") or 0.0)
            total_cost += cost
            print(
                f"  {GREEN}ok{RESET}   {name:<24} {spec.endpoint}\n"
                f"         {DIM}status={body.get('status_code')} results={count} "
                f"cost=${cost:.4f} time={body.get('time')}{RESET}"
            )
    finally:
        client.close()

    print()
    if failures:
        print(f"{RED}{failures} of {len(selected)} endpoint(s) failed{RESET}")
        return 1
    print(f"{GREEN}all {len(selected)} endpoints reachable{RESET}  "
          f"{DIM}total cost ${total_cost:.4f}{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
