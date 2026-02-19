FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY core/ core/
COPY server/ server/
COPY templates/ templates/
COPY main.py .

EXPOSE 18500
CMD ["python", "main.py", "start", "--host", "0.0.0.0", "--port", "18500"]
