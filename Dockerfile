FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry with increased timeouts
ENV POETRY_HOME=/opt/poetry \
    POETRY_INSTALLER_MAX_WORKERS=10 \
    POETRY_INSTALLER_PARALLEL=false \
    POETRY_HTTP_TIMEOUT=120
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && cd /usr/local/bin \
    && ln -s /opt/poetry/bin/poetry \
    && poetry config virtualenvs.create false

# Set working directory
WORKDIR /app

# Copy Poetry files
COPY pyproject.toml poetry.lock ./

# Install dependencies with retries
RUN for i in 1 2 3; do \
        poetry install --no-interaction --no-ansi && break || sleep 5; \
    done

# Copy application code
COPY . .

# Set environment variables
ENV WEB_CONCURRENCY=2 \
    MAX_REQUESTS=0 \
    MAX_REQUESTS_JITTER=0

# Expose port
EXPOSE 5000

# Run the application
CMD ["poetry", "run", "honcho", "start", "web"] 