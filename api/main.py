from fastapi import FastAPI, HTTPException
from api.schemas import NERRequest, NERResponse, ExtractedEntity
from core.engine import nlp_engine
import ast
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BioNLP LLaMA-3 NER Microservice",
    description="Production endpoint for Unsloth-optimized LLaMA-3 clinical extraction",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/v1/extract", response_model=NERResponse)
def extract_entities(request: NERRequest):
    try:
        # 1. Execute inference using your exact Alpaca pipeline
        raw_output = nlp_engine.predict(request.text, request.max_new_tokens)
        
        # Clean up common downstream artifact trails if any exist
        clean_output = raw_output.split("]")[0] + "]" if "]" in raw_output else raw_output
        
        processed_entities = []
        
        # 2. Safely evaluate the literal string back into native Python tuples
        try:
            parsed_tuples = ast.literal_eval(clean_output)
            if isinstance(parsed_tuples, list):
                for entity_type, entity_value in parsed_tuples:
                    processed_entities.append(
                        ExtractedEntity(entity_type=entity_type, value=entity_value)
                    )
        except Exception as parse_err:
            logger.warning(f"Fallback parsing required. Safe evaluation failed on string: {clean_output}. Error: {str(parse_err)}")
            # Fallback placeholder if generation structure degrades
            processed_entities.append(
                ExtractedEntity(entity_type="RAW_GENERATION", value=clean_output)
            )

        return NERResponse(
            status="success",
            entities=processed_entities,
            raw_output=raw_output
        )
        
    except Exception as e:
        logger.error(f"Inference Engine Crash: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal inference failure: {str(e)}")
