"""End-to-end check against a running server.

Creates a profile, runs the DAG, then exercises every read endpoint and the recheck
path, asserting status codes and response invariants. Exits non-zero on the first
failure, so it works as a post-deploy gate as well as a manual demo.

    make run          # in one terminal
    make smoke        # in another
"""

import os
import sys

import httpx

BASE = os.environ.get("BASE", "http://127.0.0.1:8000")
GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"

PROFILE_BODY = {
    "name": "Surfer SEO",
    "domain": "surferseo.com",
    "industry": "SEO Software",
    "description": "AI-powered SEO content optimization tool",
    "competitors": ["clearscope.io", "marketmuse.com", "frase.io"],
}

failures: list[str] = []


def ok(label: str, detail: str = "") -> None:
    print(f"  {GREEN}ok{RESET}   {label}" + (f" {DIM}{detail}{RESET}" if detail else ""))


def bad(label: str, detail: str) -> None:
    print(f"  {RED}FAIL{RESET} {label} - {detail}")
    failures.append(label)


def expect(want: int, response: httpx.Response, label: str) -> None:
    if response.status_code == want:
        ok(label, f"({want})")
    else:
        bad(label, f"expected {want}, got {response.status_code}: {response.text[:120]}")


def check(condition: bool, label: str, detail: str = "") -> bool:
    ok(label, detail) if condition else bad(label, detail or "assertion failed")
    return condition


def main() -> int:
    client = httpx.Client(base_url=BASE, timeout=120.0)
    print(f"→ {BASE}\n")

    print("service")
    expect(200, client.get("/healthz"), "GET  /healthz")
    expect(200, client.get("/"), "GET  /")
    unmatched = client.get("/api/v1/nope")
    expect(404, unmatched, "GET  /api/v1/nope")
    check(
        set(unmatched.json()) == {"error", "detail"},
        "404 uses the standard error envelope",
    )

    print("\nprofiles")
    created = client.post("/api/v1/profiles", json=PROFILE_BODY)
    expect(201, created, "POST /api/v1/profiles")
    if created.status_code != 201:
        return 1
    profile = created.json()["profile_uuid"]
    check(created.json()["created_at"].endswith("Z"), "created_at is UTC with a Z suffix")

    expect(409, client.post("/api/v1/profiles", json=PROFILE_BODY), "POST duplicate domain")
    expect(
        422,
        client.post("/api/v1/profiles", json={"name": "x", "domain": "localhost"}),
        "POST invalid domain",
    )
    expect(404, client.get("/api/v1/profiles/not-a-real-uuid"), "GET  unknown profile")
    expect(
        404, client.get(f"/api/v1/profiles/{profile}/queries"), "GET  queries before any run"
    )

    print(f"\npipeline run {DIM}(10-30s against live APIs; ~1s in mock mode){RESET}")
    run_response = client.post(f"/api/v1/profiles/{profile}/run", json={})
    expect(200, run_response, "POST /profiles/{uuid}/run")
    if run_response.status_code != 200:
        return 1
    run = run_response.json()

    required = {
        "run_uuid", "status", "retrieval_calls_planned", "records_normalized",
        "top_insights", "report", "token_usage",
    }
    check(required <= set(run), "response carries every field the spec requires")
    check(run["status"] in {"completed", "partial", "failed"}, "status", run["status"])
    check(run["retrieval_calls_planned"] > 0, "planner produced calls",
          str(run["retrieval_calls_planned"]))
    check(run["records_normalized"] > 0, "extraction produced records",
          str(run["records_normalized"]))
    check(bool(run["report"]["summary_markdown"]), "report has a human-readable summary")
    check(bool(run["report"]["recommendations"]), "report has recommendations",
          str(len(run["report"]["recommendations"])))
    print(f"       {DIM}tokens={run['token_usage'].get('total_tokens')} "
          f"duration={run['duration_ms']:.0f}ms degraded={run['degraded']}{RESET}")
    print(f"       {DIM}path: {' -> '.join(run['node_path'])}{RESET}")

    print("\nreads")
    expect(200, client.get(f"/api/v1/profiles/{profile}"), "GET  /profiles/{uuid}")
    queries = client.get(f"/api/v1/profiles/{profile}/queries").json()
    scores = [q["opportunity_score"] for q in queries["queries"]]
    check(scores == sorted(scores, reverse=True), "queries sorted by score desc",
          f"top={scores[0] if scores else None}")
    check(all(0.0 <= s <= 1.0 for s in scores), "scores within [0, 1]")

    expect(
        200,
        client.get(
            f"/api/v1/profiles/{profile}/queries",
            params={"min_score": 0.4, "page": 1, "per_page": 2},
        ),
        "GET  queries with filters + pagination",
    )
    expect(
        422,
        client.get(f"/api/v1/profiles/{profile}/queries", params={"status": "invisible"}),
        "GET  rejects an invalid status filter",
    )
    expect(200, client.get(f"/api/v1/profiles/{profile}/recommendations"), "GET  /recommendations")

    print("\nrecheck")
    query_uuid = queries["queries"][0]["query_uuid"]
    recheck = client.post(f"/api/v1/queries/{query_uuid}/recheck")
    expect(200, recheck, "POST /queries/{uuid}/recheck")
    check(recheck.json()["query_uuid"] == query_uuid, "recheck updates the query in place")

    after = client.get(f"/api/v1/profiles/{profile}/queries").json()
    check(
        after["run_uuid"] == run["run_uuid"],
        "recheck did not displace the latest full run",
        f"still {run['run_uuid'][:8]}",
    )

    print("\ntrace")
    trace = client.get(f"/api/v1/runs/{run['run_uuid']}").json()
    for node in trace["nodes"]:
        print(
            f"  {node['sequence']:>2}  {node['node']:<24} {node['status']:<5} "
            f"{node['duration_ms']:>9.2f}ms  retries={node['retries']}"
        )
    check(
        [n["node"] for n in trace["nodes"]] == run["node_path"],
        "trace matches the executed node path",
    )
    print(f"       {DIM}api calls: {trace['metrics']['api_calls']}{RESET}")
    print(f"       {DIM}failures : {trace['metrics']['api_failures'] or 'none'}{RESET}")

    client.close()
    print()
    if failures:
        print(f"{RED}{len(failures)} check(s) failed:{RESET} " + ", ".join(failures))
        return 1
    print(f"{GREEN}all checks passed{RESET}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except httpx.ConnectError:
        print(f"{RED}cannot reach {BASE}{RESET} - start the server first with: make run")
        sys.exit(2)
