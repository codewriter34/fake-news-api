import os
import io
import time
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
import logging

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Device — auto GPU if available ───────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Running on: {DEVICE}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

HF_REPO = "swanky237/fake-news-models"
HF_TOKEN = None


# ── Model Definitions ─────────────────────────────────────────
class MultimodalFakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        resnet = models.resnet50(pretrained=False)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(768 + 2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, input_ids, attention_mask, pixel_values=None):
        text_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_out.pooler_output
        if pixel_values is not None:
            img_feat = self.resnet(pixel_values).squeeze(-1).squeeze(-1)
        else:
            img_feat = torch.zeros(input_ids.size(0), 2048, device=DEVICE)
        combined = torch.cat([text_feat, img_feat], dim=1)
        return self.classifier(combined)


class XMRCameroonClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.xlmr = AutoModel.from_pretrained("xlm-roberta-base")
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        out = self.xlmr(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.pooler_output)


# ── Download models from HF Hub ───────────────────────────────
def download_model(filename: str):
    dest = MODELS_DIR / filename
    if dest.exists():
        logger.info(f"{filename} already on disk — skipping download")
        return
    logger.info(f"Downloading {filename} from Hugging Face Hub...")
    start = time.time()
    hf_hub_download(
        repo_id=HF_REPO,
        filename=filename,
        token=HF_TOKEN,
        local_dir=str(MODELS_DIR),
    )
    elapsed = time.time() - start
    size_mb = (MODELS_DIR / filename).stat().st_size / 1e6
    logger.info(f"{filename} downloaded — {size_mb:.1f} MB in {elapsed:.1f}s ✅")


def load_models():
    global multimodal_model, cameroon_model, bert_tokenizer, xlmr_tokenizer, eval_img_tf

    download_model("fakeddit_multimodal_evaluated_87.pth")
    download_model("xlmr_cameroon_best.pth")

    logger.info("Loading tokenizers...")
    bert_tokenizer  = BertTokenizer.from_pretrained("bert-base-uncased")
    xlmr_tokenizer  = AutoTokenizer.from_pretrained("xlm-roberta-base")

    logger.info("Loading MultimodalFakeDetector (BERT + ResNet50)...")
    multimodal_model = MultimodalFakeDetector().to(DEVICE)
    ckpt = torch.load(
        MODELS_DIR / "fakeddit_multimodal_evaluated_87.pth",
        map_location=DEVICE
    )
    multimodal_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    multimodal_model.eval()
    logger.info("MultimodalFakeDetector ready ✅")

    logger.info("Loading XMRCameroonClassifier (XLM-RoBERTa)...")
    cameroon_model = XMRCameroonClassifier().to(DEVICE)
    ckpt2 = torch.load(
        MODELS_DIR / "xlmr_cameroon_best.pth",
        map_location=DEVICE
    )
    cameroon_model.load_state_dict(ckpt2.get("model_state_dict", ckpt2))
    cameroon_model.eval()
    logger.info("XMRCameroonClassifier ready ✅")

    eval_img_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if torch.cuda.is_available():
        logger.info(f"VRAM used after loading: "
                    f"{torch.cuda.memory_allocated() / 1e9:.2f} GB / "
                    f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    logger.info("All models loaded and ready 🚀")


# Load at startup
load_models()


# ── FastAPI App ───────────────────────────────────────────────
app = FastAPI(
    title="Cameroon Fake News Detector API",
    description="Multimodal fake news detection using BERT+ResNet50 and XLM-RoBERTa, "
                "fine-tuned on Fakeddit and Cameroon fact-check data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "online",
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "models": {
            "cameroon": "XLM-RoBERTa fine-tuned on Cameroon fact-checks",
            "multimodal": "BERT + ResNet50 trained on Fakeddit dataset",
        },
        "endpoints": {
            "POST /predict": "Detect fake news from text (+ optional image)",
            "GET  /health":  "Service health check",
        }
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": True,
    }


@app.post("/predict")
async def predict(
    text: str = Form(..., description="News article text or headline to analyse"),
    image: UploadFile = File(None, description="Optional image (JPEG/PNG)"),
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")

    start_time = time.time()

    # ── Cameroon XLM-R prediction ─────────────────────────────
    try:
        enc = xlmr_tokenizer(
            text,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = cameroon_model(
                enc["input_ids"].to(DEVICE),
                enc["attention_mask"].to(DEVICE),
            )
            probs = torch.softmax(logits, dim=1)[0].tolist()

        cameroon_result = {
            "verdict":    "Fake" if probs[1] > probs[0] else "Real",
            "confidence": round(max(probs) * 100, 2),
            "fake_prob":  round(probs[1] * 100, 2),
            "real_prob":  round(probs[0] * 100, 2),
            "model":      "XLM-RoBERTa (Cameroon fine-tuned)",
        }
    except Exception as e:
        logger.error(f"Cameroon model error: {e}")
        cameroon_result = {"error": str(e)}

    # ── Multimodal BERT + ResNet prediction ───────────────────
    try:
        enc2 = bert_tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        pixel_values = None
        has_image = False

        if image and image.filename:
            contents = await image.read()
            if contents:
                img = Image.open(io.BytesIO(contents)).convert("RGB")
                pixel_values = eval_img_tf(img).unsqueeze(0).to(DEVICE)
                has_image = True

        with torch.no_grad():
            logits2 = multimodal_model(
                enc2["input_ids"].to(DEVICE),
                enc2["attention_mask"].to(DEVICE),
                pixel_values,
            )
            probs2 = torch.softmax(logits2, dim=1)[0].tolist()

        multimodal_result = {
            "verdict":    "Fake" if probs2[1] > probs2[0] else "Real",
            "confidence": round(max(probs2) * 100, 2),
            "fake_prob":  round(probs2[1] * 100, 2),
            "real_prob":  round(probs2[0] * 100, 2),
            "image_used": has_image,
            "model":      "BERT + ResNet50 (Fakeddit trained)",
        }
    except Exception as e:
        logger.error(f"Multimodal model error: {e}")
        multimodal_result = {"error": str(e)}

    elapsed_ms = round((time.time() - start_time) * 1000, 1)

    # ── Combined verdict ──────────────────────────────────────
    # Average both model fake probabilities for a combined score
    combined_fake = None
    if "fake_prob" in cameroon_result and "fake_prob" in multimodal_result:
        combined_fake = round(
            (cameroon_result["fake_prob"] + multimodal_result["fake_prob"]) / 2, 2
        )

    return JSONResponse({
        "success": True,
        "text_analysed": text[:200] + "..." if len(text) > 200 else text,
        "cameroon_model":   cameroon_result,
        "multimodal_model": multimodal_result,
        "combined_fake_probability": combined_fake,
        "overall_verdict": (
            "Fake" if combined_fake and combined_fake > 50 else "Real"
        ) if combined_fake is not None else None,
        "inference_time_ms": elapsed_ms,
        "device_used": str(DEVICE),
    })
