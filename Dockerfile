FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV EMB_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
EXPOSE 8000
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
