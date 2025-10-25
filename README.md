# 🧠 SadTalker GPU API

A GPU-accelerated FastAPI microservice that generates talking-head videos from static images and audio using SadTalker.

---

## 🧩 Sample

<video  src="docs/ORT2266_output.mp4" type="video/mp4" width='100%'> </video>

---

## 🚀 Features

- Full GPU inference (no CPU fallback)
- FastAPI server with `/v1/talking-head` endpoint
- Token-based authentication via `Authorization: Bearer <token>`
- Mounted data + checkpoints volumes
- GPU-enabled Docker Compose setup
- Health check endpoint `/health`

---

## 🧩 Environment Variables

Create a `.env` file in the project root with:

```bash
TALKING_HEAD_SERVICE_TOKEN=my-secret-token
```

---

## 🐋 Docker Compose Setup

Save as `docker-compose.yml`:

```yaml
version: "3.8"

services:
  sadtalker:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: sadtalker-gpu
    restart: unless-stopped

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
      - MODELS_DIR=/models
      - TALKING_HEAD_SERVICE_TOKEN=[my-secret-token]

    volumes:
      - ./models:/models
      - ./data:/app/data

    ports:
      - "2266:2266"

    # ✅ Added pre-start permission fix here
    entrypoint: >
      bash -c "
      echo '🔧 Fixing directory permissions...' &&
      chown -R 1000:1000 /app/data /models || true &&
      chmod -R 777 /app/data /models || true &&
      if [ ! -d /models/checkpoints ]; then
        echo '🔽 Models not found — downloading SadTalker pretrained weights...';
        cd /app && python3 scripts/download_models.py --path /models || true;
      fi &&
      echo '🚀 Starting SadTalker FastAPI server...' &&
      uvicorn server:app --host 0.0.0.0 --port 2266
      "

    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:2266/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 20s
```

---

## 🧱 Build and Run

```bash
docker compose up --build
```

Once running:
- API available at: **http://localhost:2266**
- Swagger docs at: **http://localhost:2266/docs**

Use the **Authorize** button in Swagger UI and enter:

```
Bearer my-secret-token
```

---

## 🧪 Example Request (via curl)

```bash
curl -X POST http://localhost:2266/v1/talking-head \
  -H "Authorization: Bearer my-secret-token" \
  -F "jobId=job001" \
  -F "image=@/path/to/image.png" \
  -F "audio=@/path/to/audio.wav" \
  -o output.mp4
```

---

## ✅ Health Check

```bash
curl http://localhost:2266/health
```

Response:
```json
{"status": "ok", "timestamp": "2025-10-26T12:00:00Z"}
```

---

## 📁 Directory Structure

```
/app
 ├── server.py
 ├── inference.py
 ├── data/
 │   └── results/
 ├── checkpoints/
 ├── Dockerfile
 ├── docker-compose.yml
 └── .env
```

---

## ⚙️ Notes

- Ensure `checkpoints/` folder contains all SadTalker pretrained weights (e.g., `epoch_20.pth`).
- Requires NVIDIA GPU with CUDA 11+.
- Default API token fallback is `"my-secret-token"` if `.env` missing.

---

## 🧑‍💻 Author

Asrix AI Labs – AI Video & Avatar Systems  
**Website:** [https://asrix.com](https://asrix.com)

