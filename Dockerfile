# ==============================================================
# MCP AI Assistant — Dockerfile
# ==============================================================
# Multi-stage build: installs all Python dependencies, then
# creates a slim production image.
# ==============================================================

# ---- Builder stage ----
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system build deps (needed for sentence-transformers / numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ---- Production stage ----
FROM python:3.11-slim AS production

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/feedback data/conversations results

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default: run the REST API server
# Override with --disable-auth to work without API keys during testing
CMD ["python", "src/mcp_server/rest_api.py", "--host", "0.0.0.0", "--port", "8000"]
