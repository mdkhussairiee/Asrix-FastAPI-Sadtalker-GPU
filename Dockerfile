# ===========================
# SadTalker GPU (Production)
# ===========================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODELS_DIR=/models

# --- System dependencies ---
RUN apt-get update && apt-get install -y \
    git wget curl ffmpeg libsm6 libxext6 libgl1 python3 python3-pip unzip \
    && rm -rf /var/lib/apt/lists/*

# --- Python & PyTorch ---
RUN pip3 install --upgrade pip
RUN pip3 install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html

# --- Clone SadTalker repo ---
WORKDIR /app
RUN git clone https://github.com/OpenTalker/SadTalker.git .
RUN pip3 install -r requirements.txt
RUN pip3 install fastapi uvicorn onnxruntime-gpu gfpgan basicsr==1.4.2 python-multipart

# ===============================
# ✅ Download pretrained weights
# ===============================
RUN set -eux; \
    mkdir -p $MODELS_DIR/checkpoints $MODELS_DIR/gfpgan/weights; \
    cd $MODELS_DIR/checkpoints; \
    for url in \
        https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar \
        https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar \
        https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors \
        https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors \
        https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/epoch_20.pth \
        https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/shape_predictor_68_face_landmarks.dat \
        https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/facevid2vid_00189-model.pth.tar \
        https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/BFM_Fitting.zip; \
    do \
        echo "⬇️ Downloading $url"; \
        wget --tries=5 --timeout=30 -nc "$url" || echo "⚠️ Warning: failed to fetch $url"; \
    done; \
    unzip -n BFM_Fitting.zip -d . || true; \
    cd $MODELS_DIR/gfpgan/weights; \
    for url in \
        https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth \
        https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth \
        https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth \
        https://huggingface.co/damaimai/parsing_parsenet.pth/resolve/main/parsing_parsenet.pth; \
    do \
        echo "⬇️ Downloading $url"; \
        wget --tries=5 --timeout=30 -nc "$url" || echo "⚠️ Warning: failed to fetch $url"; \
    done

# --- Copy FastAPI server ---
COPY server.py /app/server.py

EXPOSE 2266

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "2266"]
