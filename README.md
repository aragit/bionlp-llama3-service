
<h1 align="center">Biomedical Entity Extraction Engine</h1>

<p align="center">
  <!-- API & Runtime -->
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-3776AB.svg?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI 0.110+"></a>
  <a href="https://docs.pydantic.dev/"><img src="https://img.shields.io/badge/Pydantic-v2-e92063.svg?style=flat-square&logo=pydantic&logoColor=white" alt="Pydantic v2"></a>
  <a href="https://www.uvicorn.org/"><img src="https://img.shields.io/badge/Uvicorn-0.29%2B-4051B5.svg?style=flat-square&logo=python&logoColor=white" alt="Uvicorn 0.29+"></a>
  <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/Docker-24.0%2B-2496ED.svg?style=flat-square&logo=docker&logoColor=white" alt="Docker 24.0+"></a>
</p>
<p align="center">
  <!-- ML & Inference -->
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 2.3+"></a>
  <a href="https://huggingface.co/docs/transformers/"><img src="https://img.shields.io/badge/Transformers-4.40%2B-FFD21E.svg?style=flat-square&logo=huggingface&logoColor=black" alt="Transformers 4.40+"></a>
  <a href="https://github.com/unslothai/unsloth"><img src="https://img.shields.io/badge/Unsloth-optimized-FF6B6B.svg?style=flat-square&logo=fire&logoColor=white" alt="Unsloth Optimized"></a>
  <a href="https://github.com/TimDettmers/bitsandbytes"><img src="https://img.shields.io/badge/bitsandbytes-0.43%2B-76B900.svg?style=flat-square&logo=nvidia&logoColor=white" alt="bitsandbytes 0.43+"></a>
  <a href="https://ai.meta.com/llama/"><img src="https://img.shields.io/badge/LLaMA--3-8B-0467DF.svg?style=flat-square&logo=meta&logoColor=white" alt="LLaMA-3 8B"></a>
  <a href="https://huggingface.co/docs/peft/main/en/conceptual_guides/lora"><img src="https://img.shields.io/badge/LoRA-finetuned-8A2BE2.svg?style=flat-square&logo=layers&logoColor=white" alt="LoRA Fine-tuned"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-12.x-76B900.svg?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA 12.x"></a>
  <a href="https://www.kaggle.com/"><img src="https://img.shields.io/badge/Kaggle-Ready-20BEFF.svg?style=flat-square&logo=kaggle&logoColor=white" alt="Kaggle Ready"></a>
</p>

--- 

A production-ready, hardware-aware FastAPI microservice designed for high-throughput Biological Named Entity Recognition (NER). This architecture bridges the gap between state-of-the-art LLM capabilities and resource-constrained production environments.

By leveraging Unsloth’s Triton-optimized inference kernels and 4-bit quantization, this microservice delivers low-latency entity extraction on minimal VRAM footprints, making it ideal for deployment on consumer-grade GPUs or edge-cloud instances.

##  📋Features

- **5 Biomedical Entity Types:** DNA, RNA, protein, cell_type, cell_line
- **4-Bit Quantized Inference:** Runs LLaMA-3 8B on 8GB VRAM (T4) via `bitsandbytes` + Unsloth Triton kernels
- **Deterministic Output Parsing:** Symbolic evaluation pipeline replaces fragile regex with structured tuple extraction
- **Dual Runtime Modes:** `local` (mock engine for CI/schema testing) and `gpu` (full Unsloth inference)
- **Hardware-Aware Architecture:** Environment-aware engine initialization — no GPU required for API development
- **FastAPI + Pydantic v2:** High-speed schema validation, auto-generated OpenAPI docs, async-ready endpoints
- **Docker-Ready:** Single-command deployment with `docker-compose`



## 🏗️ Technical Philosophy
This service moves away from monolithic model serving by enforcing a decoupled architecture:

- Compute Optimization: By utilizing 4-bit bitsandbytes quantization combined with Unsloth's optimized Triton kernels, we drastically reduce the VRAM overhead required to hold the model weights, enabling 8B parameter models to run on lower-tier GPUs (like the T4) without significant performance degradation.

- API-Inference Decoupling: The architecture separates the FastAPI ingestion layer from the heavy inference chassis. The API handles schema validation and payload parsing, while the engine manages stateful model execution. This allows for horizontal scaling of the API layer independently of the compute-heavy model engine.

- Deterministic Parsing: We bypass the unreliability of standard LLM generation by forcing structured output, which is then passed through a symbol-evaluation pipeline (ast.literal_eval). This ensures that the unstructured text output of the LLM is transformed into strict, type-safe data objects before reaching the client.

## 🔬 Practical Applications
This microservice is built to power high-utility biological data processing pipelines, including:

- Automated Biomedical Literature Mining: Extracting gene, protein, and cell-line mentions from thousands of PubMed abstracts at scale for meta-analysis.

- Clinical Decision Support Systems (CDSS): Parsing unstructured electronic health records (EHR) to map symptoms and biomarkers to standardized ontologies (e.g., SNOMED-CT or MeSH).

- Knowledge Graph Construction: Powering downstream automated graph construction by identifying entities and their relations (e.g., Protein-Protein Interactions or Drug-Target affinities) directly from raw scientific text.

- Drug Discovery Pipelines: Expediting target identification by enabling high-throughput semantic search over vast repositories of unstructured pharmaceutical patent data.

---

## 🏗️ System Architecture



The system implements a **Decoupled Micro-Kernel Architecture**, separating the API ingestion layer from the heavy-compute inference engine. This design allows for rapid development on local hardware (CPU) while ensuring high-performance production deployment on GPU clusters (CUDA).

### 1. Architectural Components

* **Ingestion Layer (FastAPI + Pydantic v2):** Serves as the service entry point. Pydantic v2 provides high-speed schema validation, ensuring that inbound requests meet specific type constraints before triggering any compute-heavy operations.
* **Abstraction Engine (`BioNLPEngine`):** A Factory-pattern wrapper that abstracts the hardware interface. It handles environment-aware initialization—switching dynamically between a `MockEngine` for local testing and a `TritonEngine` for production.
* **Inference Chassis (Unsloth + Triton):** The core execution layer. By utilizing Unsloth's optimized Triton kernels, we reduce the computational overhead of the LLaMA-3 8B model, achieving near-native inference speeds on consumer-grade and cloud-provider GPUs.
* **Deterministic Parsing Layer:** Unlike traditional LLM pipelines that rely on fragile Regex, this layer uses `ast.literal_eval()` for secure, symbolic evaluation. It creates a robust bridge between the probabilistic nature of the LLM and the deterministic needs of your downstream data structures.

### 2. Runtime Environment Strategy
The system utilizes the `RUNTIME_ENV` injection variable to modify the execution stack at startup, enabling seamless transition across development and deployment cycles.

| Environment | Engine Type | Capability | Target Hardware |
|:---|:---|:---|:---|
| `local` | `MockEngine` | Functional Schema Verification | Local CPU / IDE |
| `gpu` | `TritonEngine` | 4-bit Quantized Inference | NVIDIA GPU (T4/P100/A100) |

### 3. Execution Flow
The pipeline follows a strict, unidirectional state machine flow to minimize latency and ensure data integrity:

```
[HTTP Request] 
      ↓
[Pydantic Validation (Contract Verification)]
      ↓
[BioNLPEngine (State Check: Local vs. GPU)]
      ↓
[Model Generation (LLaMA-3 8B + 4-bit Quantization)]
      ↓
[Deterministic Parsing (AST Literal Evaluation)]
      ↓
[HTTP Response (Structured JSON Object)]
```

## 🧬 Model

- Base: *LLaMA-3 8B*
- Quantization: *Unsloth 4-bit* 
- ine-tuning:  *LoRA on biomedical NER*
  *This model takes a medical or biological text as input and identifies and extracts the following five entity types:*
  *- DNA*
  *- RNA*
  *- protein*
  *- cell_type*
  *- cell_line*
  
   The output is a clean, machine-readable Python list of tuples. read more : **[Arnic/llama-3-8b-bionlp-ner](https://huggingface.co/Arnic/llama-3-8b-bionlp-ner)**
  
- nference: *Triton kernels* 

---

## Performance Benchmarks

All measurements are for **LLaMA-3 8B 4-bit quantized inference** via Unsloth Triton kernels, extracting entities from PubMed-scale biomedical abstracts (~200–400 input tokens, `max_new_tokens=128`).

### Reference Targets

> **Note:** These are hardware-specific reference targets based on Unsloth 4-bit LLaMA-3 8B community benchmarks. Run the [reproduction script](#reproducing) below to generate your own exact figures.

| Hardware | VRAM Footprint | Cold Start | Latency p50 | Latency p99 | Throughput* | Max Concurrent |
|----------|---------------|------------|-------------|-------------|-------------|----------------|
| NVIDIA T4 (16 GB) | ~6.5 GB | ~4.2 s | ~2.8 s | ~5.1 s | ~0.35 req/s | 1 (no queue) |
| NVIDIA P100 (16 GB) | ~6.5 GB | ~3.1 s | ~1.9 s | ~3.4 s | ~0.52 req/s | 1–2 |
| NVIDIA A100 40 GB | ~6.5 GB | ~1.8 s | ~1.1 s | ~2.0 s | ~0.90 req/s | 4–8 |
| CPU (Mock Engine) | ~0.5 GB | ~0.02 s | ~0.005 s | ~0.01 s | ~200 req/s | 100+ |

*\*Throughput = sustained sequential requests with `asyncio.Semaphore(1)` guarding the GPU. Concurrent throughput requires a request queue.*

### What the Numbers Mean

- **Cold Start:** Time from `BioNLPEngine.__init__()` to first successful inference (model download + CUDA kernel compilation).
- **Latency p50/p99:** End-to-end HTTP request latency including tokenization, generation, and deterministic parsing.
- **VRAM Footprint:** Peak GPU memory during inference. The model weights occupy ~4.5 GB; activations + KV cache add ~2 GB at `max_seq_length=2048`.
- **Max Concurrent:** Hard limit before CUDA OOM without implementing a request queue or batching.

### Reproducing

Save this as `benchmark.py` in your repo root and run:

```python
import time
import asyncio
import statistics
import httpx

URL = "http://localhost:8000/v1/extract"
PAYLOAD = {
    "text": "IL-2 stimulates the proliferation of T cells and upregulates the transcription of specialized RNA sequences.",
    "max_new_tokens": 128
}
WARMUP_RUNS = 3
BENCHMARK_RUNS = 50

async def main():
    async with httpx.AsyncClient() as client:
        # Warmup
        for _ in range(WARMUP_RUNS):
            await client.post(URL, json=PAYLOAD)

        # Benchmark
        latencies = []
        for _ in range(BENCHMARK_RUNS):
            t0 = time.perf_counter()
            r = await client.post(URL, json=PAYLOAD)
            r.raise_for_status()
            latencies.append(time.perf_counter() - t0)

        print(f"Runs: {BENCHMARK_RUNS}")
        print(f"p50: {statistics.median(latencies):.3f}s")
        print(f"p99: {sorted(latencies)[int(len(latencies)*0.99)]:.3f}s")
        print(f"Mean: {statistics.mean(latencies):.3f}s")
        print(f"Min/Max: {min(latencies):.3f}s / {max(latencies):.3f}s")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:

```bash
RUNTIME_ENV=gpu uvicorn api.main:app --host 0.0.0.0 --port 8000
# In another terminal:
python benchmark.py
```

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

Navigate to **http://localhost:8000/docs** for the interactive UI.

| Method | Path | Purpose |
|:---|:---|:---|
| GET | `/health` | Liveness probe |
| POST | `/v1/extract` | NER extraction |
---
![Loading](assets/bionlp_01.png)

**Pipeline:**
```
[JSON In] → [Pydantic] → [BioNLPEngine] → [Parser] → [JSON Out]
```

### **The Extraction & Parsing Pipeline**

1. **Prompt Formatting:** Uses a rigid Alpaca instruction-tuning layout to minimize hallucinations and enforce context.
2. **Token Generation:** Constrains the LLM to output predictable Python-serialized tuples, optimizing for token density.
3. **Deterministic Isolation:** Middleware scans for the terminal `]` to truncate "trailing noise" common in LLM generations.
4. **Symbolic Evaluation:** Utilizes `ast.literal_eval()` for secure, risk-free conversion of string-based literals into typed objects.
5. **Pydantic Validation:** Maps unstructured extractions into rigid data contracts, ensuring strict schema compliance before the response exits the system.

---


## 🧪 Try It

1. Open `POST /v1/extract` in docs
2. Click **Try it out**
3. Paste:
```json
{
  "text": "IL-2 stimulates the proliferation of T cells and upregulates the transcription of specialized RNA sequences.",
  "max_new_tokens": 128
}
```
4. Click **Execute**
---
![Docs](assets/bionlp_02.png)

---
## 📸 Interface Gallery

| Initial Loading | Schema Interaction | Extraction Result |
|:---:|:---:|:---:|
| ![Loading](assets/bionlp_01.png) | ![Docs](assets/bionlp_02.png) | ![Result](assets/bionlp_03.png) |


## Target Production Deployment (Kaggle)
To process live biomedical literature with hardware acceleration via Unsloth's optimized Triton kernels:

1. Notebook Configuration

Accelerator: Set to GPU T4 x2 (or P100) in the right-hand sidebar.

Internet: Ensure the Internet toggle in the settings sidebar is set to On.

2. Environment Provisioning

```Python
!git clone https://github.com/aragit/bionlp-llama3-service.git
%cd bionlp-llama3-service
!pip install -r requirements-gpu.txt
```
3. Execution Launch

```Python
# Launch the inference gateway
!RUNTIME_ENV=gpu uvicorn api.main:app --host 0.0.0.0 --port 8000
```

*Note: Because Kaggle notebooks run in a container, the API will be accessible internally. To access this API from your local machine, you will need to tunnel the connection using a utility like ngrok.*

---

## 📋 Requirements & Prerequisites

### Prerequisites
Before installing the dependencies, ensure your environment meets the following specifications:

* **Python:** Version 3.10 or higher.
* **GPU Drivers (GPU Mode Only):** NVIDIA drivers with CUDA 12.x support installed.
* **Memory (GPU Mode Only):** Minimum 16GB RAM and a GPU with at least 8GB VRAM (T4/P100 recommended).

### Dependency Installation
The project uses environment-specific dependency files to manage the split between lightweight API testing and resource-heavy model execution.

| File | Use Case | Installation Command |
|:---|:---|:---|
| `requirements-api.txt` | Local testing, schema dev, CI/CD | `pip install -r requirements-api.txt` |
| `requirements-gpu.txt` | Production inference, Unsloth, Triton | `pip install -r requirements-gpu.txt` |

> **Note:** When deploying to production (Kaggle/Cloud), always use the `requirements-gpu.txt` file. This includes `bitsandbytes` and `triton` kernels required for 4-bit quantization which are not supported on standard CPU-only environments.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `CUDA out of memory` on T4 | Concurrent requests or long input text | Reduce `max_new_tokens`, add request queue, or use `torch.cuda.empty_cache()` |
| `Model not found` on startup | `Arnic/llama-3-8b-bionlp-ner` not cached | First run downloads ~5GB; ensure stable internet or pre-download in Dockerfile |
| Returns `mock_receptor` entities | `RUNTIME_ENV` defaults to `local` | Set `RUNTIME_ENV=gpu` before starting Uvicorn |
| `ast.literal_eval` parse failure | LLM generated malformed output | Check `raw_output` field in response; ensure input text is under token limit |
| Slow first request | CUDA cold start / kernel compilation | Send a warmup request on startup or use `CUDA_LAUNCH_BLOCKING=0` |



