#!/bin/sh
# Container entrypoint: fetch the siblings for this app branch, then serve.
#
# Fetching here (not at image build time) means every restart/redeploy runs the
# LATEST commit of the matching sibling branch — siblings are maintained first,
# and a sibling-only change needs no app rebuild, just a restart. On Railway the
# branch comes from $RAILWAY_GIT_BRANCH (or $SIBLING_BRANCH if you set it).
# Fetch failures are not fatal: the affected page says its sibling is missing.
set -e
python scripts/siblings.py fetch || true
python scripts/siblings.py data || true
exec streamlit run Home.py --server.port "${PORT:-8501}" --server.address 0.0.0.0 --server.headless true
