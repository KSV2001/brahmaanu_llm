FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Copy only what’s needed
COPY brahmaanu_llm/ brahmaanu_llm/
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# Cache dir for HF models (mount a volume here on RunPod)
ENV HF_HOME=/cache/hf \
    HF_HUB_CACHE=/cache/hf \
    TRANSFORMERS_CACHE=/cache/hf

# Gradio port
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=8080

EXPOSE 8080

# Healthcheck (Gradio root returns 200 once up)
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8080/ || exit 1

CMD ["python", "-m", "brahmaanu_llm.app.main_gradio"]
