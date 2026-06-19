from pydantic import BaseModel, Field
from typing import List

class NERRequest(BaseModel):
    text: str = Field(..., description="Unstructured clinical text (e.g., discharge summary).")
    max_new_tokens: int = Field(default=256, le=1024, description="Generation limit.")

class ExtractedEntity(BaseModel):
    entity_type: str = Field(..., description="e.g., DISEASE, MEDICATION, BIOMARKER")
    value: str = Field(..., description="The extracted span of text.")

class NERResponse(BaseModel):
    status: str = "success"
    entities: List[ExtractedEntity]
    raw_output: str = Field(None, description="The raw string output from the LLM before parsing.")