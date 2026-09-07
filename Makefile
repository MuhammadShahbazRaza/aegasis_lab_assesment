VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: install run demo demo-failure smoke check-dataforseo test lint typecheck clean

install:
	$(PY) -m pip install -q --upgrade pip
	$(PIP) install -q -e ".[dev]"
	@test -f .env || cp .env.example .env
	@echo "installed. edit .env to add GROQ_API_KEY, or leave LLM_PROVIDER=fake to run offline."

run:
	$(VENV)/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

demo:
	$(PY) scripts/demo.py

# Takes three of the five endpoints down, which drops retrieval below
# RETRIEVAL_SUCCESS_THRESHOLD and forces the partial_data_fallback route.
demo-failure:
	MOCK_ALWAYS_FAIL_TOOLS=serp_organic_results,ai_overview_snapshot,llm_answer_visibility \
		$(PY) scripts/demo.py

# End-to-end check against a server already running on :8000 (make run).
smoke:
	$(PY) scripts/smoke.py

# Verifies DataForSEO credentials against whatever DATAFORSEO_BASE_URL points at.
# Point it at the sandbox first: it is free and returns real response envelopes.
check-dataforseo:
	$(PY) scripts/check_dataforseo.py

test:
	$(VENV)/bin/pytest -q

lint:
	$(VENV)/bin/ruff check app tests scripts

typecheck:
	$(VENV)/bin/mypy app

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache search_intel.db runs
	find . -name __pycache__ -type d -not -path "./.venv/*" -exec rm -rf {} +
