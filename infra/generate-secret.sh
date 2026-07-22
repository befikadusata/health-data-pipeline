#!/usr/bin/env bash
# Generates a random AIRFLOW_SECRET_KEY into infra/.env, so this checkout's
# Airflow webserver session/cookie signing doesn't rely on the placeholder
# value docker-compose.yml falls back to (which every clone of this repo
# shares). Run once, before `docker compose up`.
#
# infra/.env is tracked in git (see its own comment) - do not commit the
# generated line. If you fork or share this repo, remove it first.
set -euo pipefail
cd "$(dirname "$0")"

if grep -q '^AIRFLOW_SECRET_KEY=' .env 2>/dev/null; then
  echo "AIRFLOW_SECRET_KEY already set in infra/.env - leaving as is."
  exit 0
fi

echo "AIRFLOW_SECRET_KEY=$(openssl rand -hex 32)" >> .env
echo "Generated a random AIRFLOW_SECRET_KEY in infra/.env (do not commit this line)."
