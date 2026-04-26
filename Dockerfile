# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working dir ───────────────────────────────────────────────────────────────
WORKDIR /app

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source ───────────────────────────────────────────────────────────────
COPY models.py     .
COPY client.py     .
COPY openenv.yaml  .
COPY server/       ./server/

# ── Non-root user (HF Spaces security requirement) ────────────────────────────
RUN useradd -m -u 1000 user && chown -R user /app
USER user

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Entry — run the OpenEnv FastAPI server ────────────────────────────────────
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
