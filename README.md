# Cameroon Fake News Detector API

Multimodal fake news detection API using:
- **XLM-RoBERTa** fine-tuned on Cameroon fact-check data
- **BERT + ResNet50** trained on the Fakeddit multimodal dataset

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info and status |
| GET | `/health` | Health check |
| POST | `/predict` | Detect fake news |

### POST /predict

**Form fields:**
- `text` (required) — news article text or headline
- `image` (optional) — image file (JPEG or PNG)

**Example response:**
```json
{
  "success": true,
  "cameroon_model": {
    "verdict": "Fake",
    "confidence": 91.4,
    "fake_prob": 91.4,
    "real_prob": 8.6,
    "model": "XLM-RoBERTa (Cameroon fine-tuned)"
  },
  "multimodal_model": {
    "verdict": "Fake",
    "confidence": 87.2,
    "fake_prob": 87.2,
    "real_prob": 12.8,
    "image_used": false,
    "model": "BERT + ResNet50 (Fakeddit trained)"
  },
  "combined_fake_probability": 89.3,
  "overall_verdict": "Fake",
  "inference_time_ms": 142.3,
  "device_used": "cuda"
}
```

---

## Calling from JavaScript (your web app)

```javascript
async function detectFakeNews(text, imageFile = null) {
  const formData = new FormData();
  formData.append('text', text);
  if (imageFile) formData.append('image', imageFile);

  const response = await fetch('https://YOUR_RUNPOD_URL/predict', {
    method: 'POST',
    body: formData,
  });

  const result = await response.json();
  return result;
}

// Usage
const result = await detectFakeNews("Breaking: Scientists discover cure for cancer");
console.log(result.overall_verdict);      // "Fake" or "Real"
console.log(result.combined_fake_probability); // e.g. 89.3 (%)
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Hugging Face access token (write access) |
| `HF_REPO` | HF repo containing your .pth model files |

---

## Local Development

```bash
cp .env.example .env
# Fill in HF_TOKEN and HF_REPO in .env

docker-compose up --build
# API available at http://localhost:8080
```

## RunPod Deployment

See deployment guide in project docs.
