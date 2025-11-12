# PyTorch 2.8.0 built for CUDA 12.1 + cuDNN 9 
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
RUN pip install --upgrade torch==2.8.0 torchvision==0.23.0

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && rm -rf /var/lib/apt/lists/*


# Copy only what’s needed
COPY requirements.txt /app/requirements.txt
COPY app/ app/
COPY configs/ configs/
COPY rag/ rag/
COPY data/raw/docs/ data/raw/docs/


RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Cache dir for HF models (mount a volume here on RunPod)
ENV HF_HOME=/cache/hf \
    HF_HUB_CACHE=/cache/hf \
    TRANSFORMERS_CACHE=/cache/hf


# Serverless API port
ENV API_HOST=0.0.0.0 \
    API_PORT=7861

EXPOSE 7861
EXPOSE 8080

# HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
#   CMD curl -fsS http://127.0.0.1:7861/sample_questions || exit 1

CMD ["python", "-m", "app.api"]


# # Gradio port
# ENV GRADIO_SERVER_NAME=0.0.0.0 \
#     GRADIO_SERVER_PORT=8080

# EXPOSE 8080

# # Healthcheck (Gradio root returns 200 once up)
# HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
#   CMD curl -fsS http://127.0.0.1:8080/ || exit 1

# # Default startup command
# CMD ["python", "-m", "app.main_gradio"]

