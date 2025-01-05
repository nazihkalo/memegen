ARG ARG_PORT=5000
ARG ARG_MAX_REQUESTS=0
ARG ARG_MAX_REQUESTS_JITTER=0

# Use bookworm (Debian 12) which has SQLite 3.40.1
FROM docker.io/python:3.12.5-bookworm as build

# Install system dependencies
RUN apt update && apt install --yes webp cmake sqlite3
RUN pip install poetry

# Set the working directory
WORKDIR /app

# Copy project files
COPY . .

# Install project dependencies
RUN poetry install --no-dev

# Set environment variables
ENV PATH="/app/.local/bin:${PATH}"
ENV PORT="${ARG_PORT:-5000}"
ENV MAX_REQUESTS="${ARG_MAX_REQUESTS:-0}"
ENV MAX_REQUESTS_JITTER="${ARG_MAX_REQUESTS_JITTER:-0}"

# Set the entrypoint
ENTRYPOINT poetry run gunicorn --bind "0.0.0.0:$PORT" \
    --worker-class uvicorn.workers.UvicornWorker  \
    --max-requests="$MAX_REQUESTS" \
    --max-requests-jitter="$MAX_REQUESTS_JITTER" \
    --timeout=20  \
    app.main:app
