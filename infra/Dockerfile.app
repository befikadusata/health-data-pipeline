# Shared image for the API and the one-off migration job. The full project
# (data_gen/warehouse/validation/models/api) is bind-mounted over this at
# runtime in docker-compose.yml, so the COPY below just makes the image
# runnable standalone (e.g. `docker run` outside compose).
FROM python:3.12-slim

WORKDIR /app

COPY infra/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app

# Non-root by default for standalone `docker run`. docker-compose.yml
# overrides this at runtime (user: "${AIRFLOW_UID:-1000}:0") to match the
# host UID instead, same reasoning as infra/Dockerfile.airflow - see
# infra/.env's comment.
RUN useradd --create-home --uid 1000 app
USER app

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
