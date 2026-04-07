FROM python:3.12-slim

WORKDIR /app

# Install system deps for scipy/numpy
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir uv && \
    uv pip install --system ".[app]"

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app/slm_app.py", "--server.address=0.0.0.0", "--server.headless=true"]
