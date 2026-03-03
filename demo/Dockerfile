FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY core/ core/
COPY cli/ cli/
COPY server/ server/
COPY templates/ templates/
COPY main.py .

RUN pip install --no-cache-dir ".[rag]"

COPY demo/presets/ /demo/presets/
COPY demo/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 18500

ENTRYPOINT ["/entrypoint.sh"]
