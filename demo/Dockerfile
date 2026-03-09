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

ENV ANIMAWORKS_TEMPLATES_DIR=/app/templates

COPY demo/presets/ /demo/presets/
COPY demo/examples/ /demo/examples/
COPY demo/adjust_dates.sh /demo/adjust_dates.sh
COPY demo/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh /demo/adjust_dates.sh

EXPOSE 18501

ENTRYPOINT ["/entrypoint.sh"]
