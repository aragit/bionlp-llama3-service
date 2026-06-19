# BioNLP LLaMA-3 NER Microservice

A production-ready, hardware-aware FastAPI microservice for Biological Named Entity Recognition (NER). Exposes an inference gateway for a fine-tuned LLaMA-3 8B model with Unsloth 4-bit quantization.

---

## 🏗️ Architecture

| Mode | Variable | Behavior |
|:---|:---|:---|
| Local Blueprint | `RUNTIME_ENV=local` | Emulates output for API testing without GPU |
| GPU Production | `RUNTIME_ENV=gpu` | Boots Unsloth, loads 4-bit model, binds CUDA |

```
├── api/
│   ├── main.py      # FastAPI app
│   └── schemas.py   # Pydantic contracts
├── core/
│   └── engine.py    # Inference abstraction
├── requirements-api.txt
└── requirements-gpu.txt
```

---

## 💻 Local Setup

**1. Environment**
```bash
cd ~/bionlp-llama3-unsloth
python -m venv venv
source venv/bin/activate
pip install -r requirements-api.txt
```

**2. Launch**
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
App binds to `http://0.0.0.0:8000`.

---

## 🔬 API Usage

Navigate to `http://localhost:8000/docs` for the interactive UI.

| Method | Path | Purpose |
|:---|:---|:---|
| GET | `/health` | Liveness probe |
| POST | `/v1/extract` | NER extraction |
![Loading](assets/bionlp_01.png)

**Pipeline:**
```
[JSON In] → [Pydantic] → [BioNLPEngine] → [Parser] → [JSON Out]
```

**Steps:**
1. Prompt formatting (Alpaca layout)
2. Token generation (structured tuples)
3. Truncation at terminal `]`
4. `ast.literal_eval()` for safe parsing
5. Pydantic validation

---

## 🧪 Try It

1. Open `POST /v1/extract` in docs
2. Click **Try it out**
3. Paste:
```json
{
  "text": "IL-2 stimulates T cell proliferation.",
  "max_new_tokens": 128
}
```
4. Click **Execute**
![Docs](assets/bionlp_02.png)

---
## 📸 Interface Gallery

| Initial Loading | Schema Interaction | Extraction Result |
|:---:|:---:|:---:|
| ![Loading](assets/bionlp_01.png) | ![Docs](assets/bionlp_02.png) | ![Result](assets/bionlp_03.png) |

## 🚀 Kaggle Deploy

**1. Attach GPU T4 x2 or P100**

**2. Setup:**
```python
!git clone https://github.com/aragit/bionlp-llama3-unsloth.git
%cd bionlp-llama3-unsloth
!pip install -r requirements-gpu.txt
```

**3. Launch:**
```python
!RUNTIME_ENV=gpu uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## 📦 Requirements

| File | Use | Command |
|:---|:---|:---|
| `requirements-api.txt` | Local testing | `pip install -r requirements-api.txt` |
| `requirements-gpu.txt` | GPU production | `pip install -r requirements-gpu.txt` |

---

## 🧬 Model

| Attribute | Value |
|:---|:---|
| Base | LLaMA-3 8B |
| Quantization | Unsloth 4-bit |
| Fine-tuning | LoRA on biomedical NER |
| Inference | Triton kernels |
