FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE main.py ./
COPY core/ core/
COPY cli/ cli/
COPY server/ server/
COPY templates/ templates/

RUN pip install --no-cache-dir ".[neo4j]"

EXPOSE 18500

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:18500/api/system/health')" || exit 1

CMD ["python", "main.py", "start", "--host", "0.0.0.0", "--port", "18500"]
