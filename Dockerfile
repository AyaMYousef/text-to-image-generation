FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*


RUN curl -fsSL https://ollama.com/install.sh | sh


WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose FastAPI port
EXPOSE 8000
EXPOSE 7860

# Run Ollama in the background, wait for it, then pull llama3 model and start FastAPI
CMD /bin/sh -c "\
    ollama serve & \
    sleep 10 && \
    ollama pull llama3.2 && \
    uvicorn xray_inference_api:app --host 0.0.0.0 --port 8000 & \
    python gui.py"
