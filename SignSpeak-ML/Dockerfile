FROM python:3.10-slim

# llama.cpp requires build tools
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app/requirements.txt /app/requirements.txt

ENV CMAKE_ARGS="-DLLAMA_CUBLAS=off -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
ENV MAKEFLAGS="-j4"

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
