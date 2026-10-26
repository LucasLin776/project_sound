FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 BOLT_PROJECT_DIR=/app
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY third_party/ ./third_party/
RUN python -m pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -e '.[api]'
RUN mkdir -p checkpoints artifacts results
EXPOSE 8000
CMD ["python", "-m", "bolt_audio", "serve", "--host", "0.0.0.0", "--port", "8000"]
