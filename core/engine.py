import os
import logging

logger = logging.getLogger(__name__)

# Toggle this variable via terminal or Docker when deploying
RUNTIME_ENV = os.getenv("RUNTIME_ENV", "local") 

class BioNLPEngine:
    def __init__(self, model_name="Arnic/llama-3-8b-bionlp-ner"):
        self.model_name = model_name
        self.alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are an expert in medical text analysis. Your task is to identify and extract specific biological entities from the given text. The entity types to extract are: DNA, RNA, protein, cell_type, and cell_line.

### Input:
{}

### Response:
"""
        
        if RUNTIME_ENV == "gpu":
            logger.info("Initializing GPU Unsloth Engine...")
            self._init_gpu_engine()
        else:
            logger.info("Initializing Local Mock Engine for Blueprint Testing...")
            self._init_mock_engine()

    def _init_gpu_engine(self):
        from unsloth import FastLanguageModel
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_name,
            max_seq_length = 2048,
            dtype = None,
            load_in_4bit = True,
        )
        FastLanguageModel.for_inference(self.model)
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def _init_mock_engine(self):
        # Placeholder for local API testing without downloading models or requiring GPUs
        self.model = None
        self.tokenizer = None

    def predict(self, text: str, max_new_tokens: int = 128) -> str:
        if RUNTIME_ENV == "local":
            # Return a perfect mock string to validate your FastAPI parsing logic locally
            return "[('protein', 'mock_receptor'), ('cell_type', 'mock_lymphocyte')]"
            
        # GPU Execution Logic
        prompt = self.alpaca_prompt.format(text)
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            use_cache=True,
            do_sample=False,
            eos_token_id=self.terminators
        )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return decoded.split("### Response:")[-1].strip()

nlp_engine = BioNLPEngine()
