# Local development image. Academic Gurobi licenses are host-locked —
# prefer running on the licensed Mac with a venv. For containers, use a
# WLS (Web License Service) token via GRB_WLSACCESSID / GRB_WLSSECRET / GRB_WLSLICENSEID.
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
