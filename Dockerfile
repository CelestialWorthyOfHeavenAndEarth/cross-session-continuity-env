# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# System deps for sandbox subprocess execution
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working dir ───────────────────────────────────────────────────────────────
WORKDIR /app

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements.txt .
# Install CPU-only torch first (saves 3 GB vs full torch in container)
RUN pip install --no-cache-dir torch==2.2.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
        openenv \
        transformers>=4.40.0 \
        trl>=0.8.6 \
        datasets>=2.19.0 \
        accelerate>=0.29.0 \
        wandb>=0.17.0 \
        pytest>=8.1.0 \
        matplotlib>=3.9.0 \
        numpy>=1.26.0 \
        pandas>=2.2.0 \
        scipy>=1.13.0 \
        gradio>=4.0.0 \
        fastapi uvicorn

# ── Copy source ───────────────────────────────────────────────────────────────
COPY server/     ./server/
COPY client/     ./client/
COPY evals/      ./evals/
COPY plots/      ./plots/
COPY openenv.yaml .
COPY app.py      .

# ── Non-root user (security) ──────────────────────────────────────────────────
RUN useradd -m -u 1000 envuser && chown -R envuser /app
USER envuser

# ── Ports ─────────────────────────────────────────────────────────────────────
# 7860 = Gradio Space   8000 = MCP server (internal)
EXPOSE 7860 8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s \
    CMD curl -f http://localhost:7860/ || exit 1

# ── Entry ─────────────────────────────────────────────────────────────────────
CMD ["python", "app.py"]
