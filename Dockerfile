FROM python:3.12-slim-bookworm AS builder
WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt
FROM python:3.12-slim-bookworm AS runtime
ENV PATH="/opt/venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN groupadd -g 10001 app && useradd -u 10001 -g app -m app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=app:app app app
COPY --chown=app:app ml ml
COPY --chown=app:app web web
COPY --chown=app:app data data
COPY --chown=app:app docs docs
RUN mkdir -p models && chown app:app models docs data
USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
