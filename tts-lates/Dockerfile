FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

WORKDIR /workspace

# Cài Python + các gói cần thiết
RUN apt-get update && apt-get install -y python3 python3-pip git && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy toàn bộ code vào container
COPY . .

# Mở port Flask app (mặc định 5003)
EXPOSE 5003

CMD ["python3", "jobs_app.py"]
