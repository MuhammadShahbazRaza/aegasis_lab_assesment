import pytest

PROFILE = {
    "name": "Surfer SEO",
    "domain": "surferseo.com",
    "industry": "SEO Software",
    "description": "AI-powered SEO content optimization tool",
    "competitors": ["clearscope.io", "marketmuse.com", "frase.io"],
}


@pytest.fixture
def created(client):
    response = client.post("/api/v1/profiles", json=PROFILE)
    assert response.status_code == 201
    return response.json()


@pytest.fixture
def ran(client, created):
    response = client.post(f"/api/v1/profiles/{created['profile_uuid']}/run")
    assert response.status_code == 200
    return created, response.json()


def test_create_profile_returns_201_and_the_documented_shape(created):
    assert set(created) == {"profile_uuid", "name", "domain", "status", "created_at"}
    assert created["status"] == "created"
    assert created["created_at"].endswith("Z")


def test_urls_and_www_are_normalized_to_bare_domains(client):
    response = client.post(
        "/api/v1/profiles",
        json={**PROFILE, "domain": "https://www.Ahrefs.com/blog", "competitors": ["HTTPS://Semrush.com/"]},
    )
    assert response.status_code == 201
    assert response.json()["domain"] == "ahrefs.com"
    detail = client.get(f"/api/v1/profiles/{response.json()['profile_uuid']}").json()
    assert detail["competitors"] == ["semrush.com"]


@pytest.mark.parametrize(
    "payload",
    [
        {"domain": "x.com"},
        {"name": "No TLD", "domain": "localhost"},
        {"name": "x", "domain": "x.com", "unexpected": 1},
        {"name": "x", "domain": "x.com", "competitors": "not-a-list"},
    ],
)
def test_invalid_profile_payloads_return_422_with_field_detail(client, payload):
    response = client.post("/api/v1/profiles", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "request validation failed"
    assert body["detail"] and "field" in body["detail"][0]


def test_duplicate_domain_returns_409(client, created):
    assert client.post("/api/v1/profiles", json=PROFILE).status_code == 409


def test_unknown_profile_returns_404(client):
    response = client.get("/api/v1/profiles/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"] == "profile not found"


def test_profile_detail_carries_summary_stats(client, ran):
    created, run = ran
    detail = client.get(f"/api/v1/profiles/{created['profile_uuid']}").json()
    stats = detail["stats"]

    assert stats["total_runs"] == 1
    assert stats["last_run_uuid"] == run["run_uuid"]
    assert stats["last_run_status"] == run["status"]
    assert stats["queries_in_last_run"] > 0
    assert 0.0 <= stats["average_opportunity_score"] <= 1.0


def test_run_response_contains_every_field_the_spec_asks_for(ran):
    _, run = ran
    assert run["status"] in {"completed", "partial", "failed"}
    assert run["retrieval_calls_planned"] > 0
    assert run["records_normalized"] > 0
    assert run["top_insights"]
    assert all(0.0 <= i["opportunity_score"] <= 1.0 for i in run["top_insights"])
    assert run["report"]["summary_markdown"]
    assert run["report"]["recommendations"]
    assert run["token_usage"]["total_tokens"] > 0
    assert run["node_path"][0] == "query_planner"


def test_run_against_a_missing_profile_returns_404(client):
    assert client.post("/api/v1/profiles/nope/run").status_code == 404


def test_run_accepts_a_custom_question(client, created):
    response = client.post(
        f"/api/v1/profiles/{created['profile_uuid']}/run",
        json={"question": "Where do we lose to Clearscope in AI answers?"},
    )
    assert response.status_code == 200
    assert response.json()["question"].startswith("Where do we lose")


def test_queries_are_sorted_by_opportunity_score_descending(client, ran):
    created, _ = ran
    body = client.get(f"/api/v1/profiles/{created['profile_uuid']}/queries").json()
    scores = [q["opportunity_score"] for q in body["queries"]]
    assert scores == sorted(scores, reverse=True)
    assert body["pagination"]["total"] == len(scores)


def test_min_score_filter(client, ran):
    created, _ = ran
    base = client.get(f"/api/v1/profiles/{created['profile_uuid']}/queries").json()
    cutoff = base["queries"][0]["opportunity_score"]
    filtered = client.get(
        f"/api/v1/profiles/{created['profile_uuid']}/queries", params={"min_score": cutoff}
    ).json()
    assert all(q["opportunity_score"] >= cutoff for q in filtered["queries"])
    assert filtered["pagination"]["total"] <= base["pagination"]["total"]


def test_status_filter(client, ran):
    created, _ = ran
    body = client.get(
        f"/api/v1/profiles/{created['profile_uuid']}/queries", params={"status": "not_visible"}
    ).json()
    assert all(q["visibility_status"] == "not_visible" for q in body["queries"])


def test_invalid_filter_values_are_rejected(client, ran):
    created, _ = ran
    url = f"/api/v1/profiles/{created['profile_uuid']}/queries"
    assert client.get(url, params={"status": "invisible"}).status_code == 422
    assert client.get(url, params={"min_score": 2}).status_code == 422
    assert client.get(url, params={"page": 0}).status_code == 422


def test_pagination_splits_the_result_set(client, ran):
    created, _ = ran
    url = f"/api/v1/profiles/{created['profile_uuid']}/queries"
    first = client.get(url, params={"page": 1, "per_page": 1}).json()
    second = client.get(url, params={"page": 2, "per_page": 1}).json()

    assert len(first["queries"]) == 1
    assert first["pagination"]["total_pages"] == first["pagination"]["total"]
    assert first["queries"][0]["query_uuid"] != second["queries"][0]["query_uuid"]


def test_queries_before_any_run_returns_404_with_a_hint(client, created):
    response = client.get(f"/api/v1/profiles/{created['profile_uuid']}/queries")
    assert response.status_code == 404
    assert "hint" in response.json()["detail"]


def test_recommendations_reference_real_queries(client, ran):
    created, _ = ran
    profile_uuid = created["profile_uuid"]
    recs = client.get(f"/api/v1/profiles/{profile_uuid}/recommendations").json()
    queries = client.get(f"/api/v1/profiles/{profile_uuid}/queries").json()
    query_ids = {q["query_uuid"] for q in queries["queries"]}

    assert recs["recommendations"]
    for rec in recs["recommendations"]:
        assert rec["target_query_uuid"] in query_ids
        assert rec["priority"] in {"high", "medium", "low"}
        assert rec["content_type"] and rec["title"] and rec["rationale"]
        assert isinstance(rec["target_keywords"], list)


def test_recheck_updates_the_query_in_place(client, ran):
    created, _ = ran
    queries = client.get(f"/api/v1/profiles/{created['profile_uuid']}/queries").json()
    target = queries["queries"][0]

    response = client.post(f"/api/v1/queries/{target['query_uuid']}/recheck")
    assert response.status_code == 200
    body = response.json()

    assert body["query_uuid"] == target["query_uuid"]
    assert body["run_uuid"] != queries["run_uuid"]
    assert body["query"]["discovered_at"] >= target["discovered_at"]


def test_recheck_on_an_unknown_query_returns_404(client):
    assert client.post("/api/v1/queries/nope/recheck").status_code == 404


def test_run_trace_exposes_node_by_node_timing(client, ran):
    _, run = ran
    trace = client.get(f"/api/v1/runs/{run['run_uuid']}").json()

    assert [n["node"] for n in trace["nodes"]] == run["node_path"]
    assert all(n["duration_ms"] >= 0 for n in trace["nodes"])
    assert trace["metrics"]["api_call_total"] == run["retrieval_calls_planned"]
    assert trace["finished_at"].endswith("Z")


def test_healthz_reports_the_active_modes(client):
    body = client.get("/healthz").json()
    assert body == {"status": "ok", "llm_provider": "fake", "dataforseo_mode": "mock"}


def test_graph_endpoint_returns_the_dag_diagram(client):
    body = client.get("/api/v1/graph").json()
    assert body["format"] == "mermaid"
    for node in ("query_planner", "retrieval_worker", "report_assembler", "no_data_fallback"):
        assert node in body["diagram"]


def test_openapi_document_is_generated(client):
    spec = client.get("/openapi.json").json()
    for path in (
        "/api/v1/profiles",
        "/api/v1/profiles/{profile_uuid}",
        "/api/v1/profiles/{profile_uuid}/run",
        "/api/v1/profiles/{profile_uuid}/queries",
        "/api/v1/profiles/{profile_uuid}/recommendations",
        "/api/v1/queries/{query_uuid}/recheck",
    ):
        assert path in spec["paths"]


def test_a_recheck_does_not_displace_the_profiles_latest_full_run(client, ran):
    created, run = ran
    profile_uuid = created["profile_uuid"]
    before = client.get(f"/api/v1/profiles/{profile_uuid}/queries").json()

    target = before["queries"][0]["query_uuid"]
    assert client.post(f"/api/v1/queries/{target}/recheck").status_code == 200

    after = client.get(f"/api/v1/profiles/{profile_uuid}/queries").json()
    assert after["run_uuid"] == run["run_uuid"]
    assert after["pagination"]["total"] == before["pagination"]["total"]

    recs = client.get(f"/api/v1/profiles/{profile_uuid}/recommendations").json()
    assert recs["recommendations"]

    stats = client.get(f"/api/v1/profiles/{profile_uuid}").json()["stats"]
    assert stats["total_runs"] == 1
    assert stats["total_rechecks"] == 1
    assert stats["last_run_uuid"] == run["run_uuid"]
    assert stats["queries_in_last_run"] == before["pagination"]["total"]


def test_root_lists_the_endpoints_instead_of_404ing(client):
    body = client.get("/").json()
    assert body["docs"] == "/docs"
    assert any("/api/v1/profiles/{profile_uuid}/run" in e for e in body["endpoints"])


def test_unmatched_routes_use_the_same_error_envelope(client):
    response = client.get("/api/v1/nope")
    assert response.status_code == 404
    assert set(response.json()) == {"error", "detail"}

    not_allowed = client.delete("/api/v1/profiles")
    assert not_allowed.status_code == 405
    assert set(not_allowed.json()) == {"error", "detail"}
