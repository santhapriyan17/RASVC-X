# RASVC-X API + chat UI.
#
#   docker compose up --build
#
# No secret is baked into the image: RASVCX_LLM_API_KEY is supplied at run
# time (docker-compose reads it from .env). Models are downloaded on first
# start into the HF cache volume, not at build time.

# ---- stage 1: build the chat UI -------------------------------------------
FROM node:20-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- stage 2: backend -------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface

WORKDIR /app

# CPU-only torch keeps the image several GB smaller than the default wheel.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0.0"

COPY pyproject.toml README.md ./
COPY src/ ./src/
# Editable install: the package stays at /app/src/rasvcx, so api/main.py
# finds the UI at /app/frontend/dist (<package>/../../../frontend/dist).
RUN pip install -e ".[research,gemini]"

COPY config/ ./config/
COPY --from=frontend /frontend/dist ./frontend/dist

RUN useradd --create-home --uid 10001 rasvcx \
    && mkdir -p /app/corpus /app/data /app/.runtime /models/huggingface \
    && chown -R rasvcx:rasvcx /app /models
USER rasvcx

EXPOSE 8000

# /ready is 200 only when every component of the configured mode passed its
# probe (knowledge base loaded, Qdrant point count matches, models loaded).
# First start downloads ~2 GB of models, hence the long start period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "rasvcx", "--host", "0.0.0.0", "--port", "8000"]
