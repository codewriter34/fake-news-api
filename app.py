import os
import io
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from transformers import (
    BertTokenizer,
    BertModel,
    AutoTokenizer,
    AutoModel
)

from huggingface_hub import hf_hub_download


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

logger.info(f"Device: {DEVICE}")


# ============================================================
# PATHS
# ============================================================

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

HF_REPO = "swanky237/fake-news-models"


# ============================================================
# MODEL DEFINITIONS
# EXACT COPY OF TRAINING MODEL
# ============================================================


class MultimodalFakeDetector(nn.Module):

    def __init__(
        self,
        bert_name="bert-base-uncased",
        num_classes=2,
        dropout=0.3,
        fusion_dim=768
    ):

        super().__init__()

        self.bert = BertModel.from_pretrained(
            bert_name
        )

        self.bert_output_dim = 768


        res = models.resnet50(
            weights=None
        )

        self.cnn = nn.Sequential(
            *list(res.children())[:-1]
        )

        self.resnet_output_dim = 2048


        self.text_proj = nn.Linear(
            self.bert_output_dim,
            fusion_dim
        )


        self.vision_proj = nn.Linear(
            self.resnet_output_dim,
            fusion_dim
        )


        self.attention_weights_linear = nn.Linear(
            fusion_dim * 2,
            fusion_dim * 2
        )


        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                fusion_dim * 2,
                num_classes
            )
        )


    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values=None
    ):


        text = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )


        cls_vec = text.last_hidden_state[:,0,:]


        t_prime = self.text_proj(
            cls_vec
        )


        if pixel_values is not None:

            img = self.cnn(
                pixel_values
            )

            img = img.view(
                img.size(0),
                -1
            )


            v_prime = self.vision_proj(
                img
            )

        else:

            v_prime = torch.zeros(
                input_ids.size(0),
                768,
                device=input_ids.device
            )


        fused = torch.cat(
            [
                t_prime,
                v_prime
            ],
            dim=1
        )


        attention = self.attention_weights_linear(
            fused
        )


        weights = torch.softmax(
            attention,
            dim=1
        )


        fused = fused * weights


        return self.head(
            fused
        )



# ============================================================
# CAMEROON MODEL
# ============================================================


class XMRCameroonClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.xlmr = AutoModel.from_pretrained(
            "xlm-roberta-base"
        )

        self.classifier = nn.Linear(
            768,
            2
        )


    def forward(
        self,
        input_ids,
        attention_mask
    ):

        output = self.xlmr(
            input_ids=input_ids,
            attention_mask=attention_mask
        )


        return self.classifier(
            output.pooler_output
        )



# ============================================================
# DOWNLOAD MODELS
# ============================================================


def download_model(filename):

    path = MODELS_DIR / filename


    if path.exists():

        logger.info(
            f"{filename} already exists"
        )

        return


    hf_hub_download(
        repo_id=HF_REPO,
        filename=filename,
        local_dir=str(MODELS_DIR)
    )



# ============================================================
# LOAD EVERYTHING
# ============================================================


def load_models():

    global multimodal_model
    global cameroon_model
    global bert_tokenizer
    global xlmr_tokenizer
    global image_transform


    download_model(
        "fakeddit_multimodal_evaluated_87.pth"
    )


    download_model(
        "xlmr_cameroon_best.pth"
    )



    bert_tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased"
    )


    xlmr_tokenizer = AutoTokenizer.from_pretrained(
        "xlm-roberta-base"
    )


    logger.info(
        "Loading multimodal model..."
    )


    multimodal_model = MultimodalFakeDetector().to(
        DEVICE
    )


    checkpoint = torch.load(
        MODELS_DIR / "fakeddit_multimodal_evaluated_87.pth",
        map_location=DEVICE
    )


    if "model_state_dict" in checkpoint:

        checkpoint = checkpoint["model_state_dict"]


    multimodal_model.load_state_dict(
        checkpoint
    )


    multimodal_model.eval()



    logger.info(
        "Loading Cameroon XLM-R..."
    )


    cameroon_model = XMRCameroonClassifier().to(
        DEVICE
    )


    checkpoint = torch.load(
        MODELS_DIR / "xlmr_cameroon_best.pth",
        map_location=DEVICE
    )


    if "model_state_dict" in checkpoint:

        checkpoint = checkpoint["model_state_dict"]


    cameroon_model.load_state_dict(
        checkpoint
    )


    cameroon_model.eval()



    image_transform = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [
                    0.485,
                    0.456,
                    0.406
                ],
                [
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ]
    )


    logger.info(
        "ALL MODELS READY"
    )



load_models()



# ============================================================
# FASTAPI
# ============================================================


app = FastAPI(
    title="Cameroon Fake News Detector",
    version="1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
def home():

    return {
        "status":"online",
        "device":str(DEVICE)
    }



@app.get("/health")
def health():

    return {
        "status":"healthy",
        "models_loaded":True
    }



@app.post("/predict")
async def predict(

    text:str = Form(...),

    image:UploadFile = File(None)

):


    if not text.strip():

        raise HTTPException(
            400,
            "Empty text"
        )


    start=time.time()



    # -----------------------------
    # Cameroon model
    # -----------------------------


    cam = xlmr_tokenizer(
        text,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )


    with torch.no_grad():

        logits = cameroon_model(
            cam["input_ids"].to(DEVICE),
            cam["attention_mask"].to(DEVICE)
        )


    probs=torch.softmax(
        logits,
        dim=1
    )[0]


    cam_result={

        "verdict":
            "Fake" if probs[1]>probs[0]
            else "Real",

        "fake_probability":
            round(float(probs[1])*100,2),

        "real_probability":
            round(float(probs[0])*100,2)

    }



    # -----------------------------
    # Multimodal model
    # -----------------------------


    tok = bert_tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )


    pixels=None

    image_used=False



    if image:

        data=await image.read()

        img=Image.open(
            io.BytesIO(data)
        ).convert("RGB")


        pixels=image_transform(
            img
        ).unsqueeze(0).to(DEVICE)


        image_used=True



    with torch.no_grad():

        logits=multimodal_model(
            tok["input_ids"].to(DEVICE),
            tok["attention_mask"].to(DEVICE),
            pixels
        )


    probs=torch.softmax(
        logits,
        dim=1
    )[0]


    multi_result={

        "verdict":
            "Fake" if probs[1]>probs[0]
            else "Real",

        "fake_probability":
            round(float(probs[1])*100,2),

        "real_probability":
            round(float(probs[0])*100,2),

        "image_used":
            image_used

    }



    combined=round(
        (
            multi_result["fake_probability"]
            +
            cam_result["fake_probability"]

        )/2,
        2
    )


    return JSONResponse(

        {

        "success":True,

        "text":
            text[:200],

        "cameroon_model":
            cam_result,

        "multimodal_model":
            multi_result,


        "combined_fake_probability":
            combined,


        "final_verdict":
            "Fake" if combined>50 else "Real",


        "inference_ms":
            round(
                (time.time()-start)*1000,
                2
            )

        }

    )