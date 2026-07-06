# Teaching App — common tasks
# Run `make` or `make help` to list available targets.
# Requires: uv (https://docs.astral.sh/uv/) and, optionally, Docker for PostgreSQL.

APP          := Home.py
PG_CONTAINER := postgres-db

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | sort \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
.PHONY: install
install:  ## Create/update the uv environment (Python 3.13) from the lockfile
	uv sync

.PHONY: setup
setup: install db-up  ## Full setup: install deps + start the PostgreSQL container
	@echo "✅ Setup complete — run 'make run' to launch the app."

.PHONY: lock
lock:  ## Re-resolve dependencies and refresh uv.lock
	uv lock

.PHONY: export-reqs
export-reqs:  ## Regenerate requirements.txt from the lockfile (pip/deploy fallback)
	uv export --no-hashes --no-dev --no-emit-project -o requirements.txt

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
.PHONY: run
run:  ## Launch the Streamlit app at http://localhost:8501
	uv run streamlit run $(APP)

.PHONY: test
test:  ## Run the test suite
	uv run pytest -q

# ---------------------------------------------------------------------------
# Database (optional — the app falls back to SQLite if Postgres is absent)
# ---------------------------------------------------------------------------
.PHONY: db-up
db-up:  ## Start PostgreSQL in Docker (creates the container on first run)
	@docker start $(PG_CONTAINER) 2>/dev/null \
	  || docker run --name $(PG_CONTAINER) \
	    -e POSTGRES_PASSWORD=mypassword \
	    -e POSTGRES_USER=myuser \
	    -e POSTGRES_DB=mydatabase \
	    -p 5432:5432 \
	    -v postgres-data:/var/lib/postgresql/data \
	    -d postgres

.PHONY: db-stop
db-stop:  ## Stop the PostgreSQL container (data is preserved)
	-docker stop $(PG_CONTAINER)

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------
.PHONY: clean
clean:  ## Remove the virtualenv and Python caches
	rm -rf .venv
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
