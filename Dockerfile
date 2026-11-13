# Teaching App — container image (used by Railway, works with plain `docker build` too).
#
# The sibling lesson repos are NOT in this repo (they are independent, git-ignored
# repos). The container clones them when it STARTS (scripts/start.sh), on the branch
# matching the app's branch (develop -> develop, main -> main; see siblings.toml),
# plus the public datasets listed there. Kaggle-only datasets are absent on purpose:
# the lessons fall back to synthetic data and say so on screen.

# --- Stage 1: build the environment ------------------------------------------
# A compiler is needed only here: `ecos` (a scikit-survival dependency) ships no
# wheel for Python 3.13 and is built from source. The final image stays slim.
FROM python:3.13-slim AS builder
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.10.11 /uv /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

# --- Stage 2: the runtime image ----------------------------------------------
FROM python:3.13-slim
# git: fetch the pinned siblings. libgomp1: OpenMP runtime xgboost needs at import.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git libgomp1 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY . .

# Railway injects $PORT and $RAILWAY_GIT_BRANCH; for a local `docker run` pass
# `-e SIBLING_BRANCH=develop` to choose the sibling branch (default: main).
ENV PORT=8501
EXPOSE 8501
CMD ["sh", "scripts/start.sh"]
