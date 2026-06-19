# BioNLP LLaMA-3 NER Microservice

A production-ready, hardware-aware FastAPI microservice optimized for Biological Named Entity Recognition (NER). This service exposes an inference gateway for a fine-tuned LLaMA-3 8B model, utilizing Unsloth for optimized 4-bit quantized execution. 

The architecture decouples the structural API routing layer from the heavy execution chassis, allowing zero-dependency schema validation locally and high-throughput execution when deployed to a GPU environment.

---

## 🏗️ Architectural Overview & Infrastructure

The repository implements an environment-aware execution strategy via the `RUNTIME_ENV` switch:

- **Local Blueprint Mode (`RUNTIME_ENV=local`):** Emulates model output to facilitate testing of API routing, payload validation, parsing pipelines, and upstream/downstream integrations without requiring local NVIDIA drivers or GPU hardware.
- **GPU Production Mode (`RUNTIME_ENV=gpu`):** Dynamically boots the Unsloth execution engine, loads the specialized 4-bit model, binds to the native CUDA runtime, and processes inference over hardware acceleration.

### Core Directory Layout
```text
├── api/
│   ├── __init__.py
│   ├── main.py          # FastAPI application, routing logic, and error handlers
│   └── schemas.py       # Pydantic V2 data contracts (Inbound/Outbound shapes)
├── core/
│   ├── __init__.py
│   └── engine.py        # Environment-aware inference abstraction (Unsloth vs Mock)
├── requirements-api.txt # Lightweight dependencies for local testing
├── requirements-gpu.txt # Heavy compute dependencies for Kaggle/Cuda deployment
└── README.md
💻 Local Installation & ExecutionTo spin up the blueprint locally to inspect schemas, validate integrations, or test application state cycles:1. Environment SetupIsolate your workspace using a virtual environment:Bashcd ~/bionlp-llama3-unsloth
pip install -r requirements-api.txt
2. Launch the GatewayStart the asynchronous server process using Uvicorn with auto-reload enabled:Bashpython -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
The application will launch and bind to http://0.0.0.0:8000.🔬 Interactive UI Documentation & Scientific DescriptionFastAPI auto-compiles standard OpenAPI specifications and exposes an interactive visual playground. Once the server is running, navigate to:Plaintexthttp://localhost:8000/docs
1. Endpoint Layout and MechanicsGET /healthPurpose: Infrastructure liveness probe.Behavior: Returns an explicit 200 OK status matrix. Used by orchestrators to monitor container health.POST /v1/extractPurpose: Principal feature extraction gateway.Payload (NERRequest): Accepts a text literal and an optional max_new_tokens configuration parameter.Response (NERResponse): Returns structured JSON housing typed entities and the immutable raw model output.[ Inbound JSON Payload ] ──> [ Pydantic Validation ] ──> [ BioNLPEngine Pipeline ]
                                                                 │
                                                                 ▼
[ Outbound JSON Object ] <── [ Deterministic Tuple Parser ] <── [ Raw Token Sequence ]
2. The Extraction & Parsing PipelineThe extraction sequence employs a deterministic structural parsing pipeline designed to bridge connectionist model behavior with symbolic data processing:Prompt Formatting: The payload text is injected into an unsloth-optimized Alpaca instruction-tuning layout, configuring the system context explicitly for deep biological token parsing.Token Generation: The underlying LLM generates a predictable, structured text serialization matching an explicitly learned grammar format: a string representation of Python literal structures (tuples inside a list).Deterministic Token Isolation: Downstream sequences can occasionally trailing-bleed during inference. The parsing middleware scans for the terminal closure sequence (]) and truncates non-deterministic trailing noise:$$\text{Output}_{\text{clean}} = \text{Sequence} \cap [0, \text{index}(`]`) + 1]$$Symbolic Evaluation: Rather than using hazardous regex engines or unprincipled string splits, the raw string is parsed using Python's Abstract Syntax Tree (ast.literal_eval). This securely validates the code structure without code execution risks, converting text literals back into typed data blocks:$$\text{ast.literal\_eval}(\text{Output}_{\text{clean}}) \longrightarrow \mathbb{L}[(\text{Entity\_Type}, \text{Value})]$$Data Contract Compliance: The resulting Python list is mapped into individual nested objects typed dynamically by Pydantic, ensuring that invalid extractions are caught gracefully before leaving the system perimeter.3. Step-by-Step UI Sandbox Execution InstructionsExpand the green POST /v1/extract route section.Click the Try it out button in the upper-right corner of the route card to open the request body for modifications.Replace the placeholder values in the JSON block with your evaluation sample. For example:JSON{
  "text": "IL-2 stimulates the proliferation of T cells and upregulates the transcription of specialized RNA sequences.",
  "max_new_tokens": 128
}
Click the large blue Execute button.Scroll down to the Server response grid to examine the structured extraction, output telemetry, and response latency vectors.🚀 Target Production Deployment (Kaggle Notebooks)To process live biomedical literature with hardware acceleration via Unsloth's optimized Triton inference kernels, deploy this repository directly to Kaggle:1. Accelerator ConfigurationCreate a new Kaggle notebook and attach the GPU T4 x2 or GPU P100 accelerator through the right-hand configuration panel.2. Environment ProvisioningClone the source code repository directly into the workspace partition and compile the designated hardware dependency list:Python!git clone [https://github.com/arashnicoomanesh/bionlp-llama3-unsloth.git](https://github.com/arashnicoomanesh/bionlp-llama3-unsloth.git)
%cd bionlp-llama3-unsloth
!pip install -r requirements-gpu.txt
3. Execution LaunchOverride the execution environment variable to instruct the underlying BioNLPEngine to construct the active Unsloth architecture, map the tensors to the CUDA core layout, and ignite the server:Python!RUNTIME_ENV=gpu uvicorn api.main:app --host 0.0.0.0 --port 8000
