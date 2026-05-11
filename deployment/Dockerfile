FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency spec first for layer caching
COPY pyproject.toml ./

# Install dependencies
RUN uv pip install --system -e "."

# Copy source
COPY src/ ./src/
COPY scripts/ ./scripts/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
