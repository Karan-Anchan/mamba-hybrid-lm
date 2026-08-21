FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MAMBA_DEVICE=cpu \
    MAMBA_ALLOWED_RATIOS=1:3 \
    MAMBA_DEFAULT_RATIO=1:3

WORKDIR /app

COPY demo/requirements.txt /app/demo/requirements.txt
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r /app/demo/requirements.txt

COPY src /app/src
COPY data/tokenizer/openwebtext.json /app/data/tokenizer/openwebtext.json
COPY demo/start.py demo/checkpoint_manifest.json /app/demo/

EXPOSE 7860
CMD ["python", "demo/start.py"]
