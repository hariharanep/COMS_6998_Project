from pydantic import BaseModel, Field
from typing import Optional

class EnhancePromptReq(BaseModel):
    prompt: str = Field(description="Provided user prompt")
    model: str = Field(description="Model Name")

    # Pydantic v2 style
    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "Enhance this prompt."
            }
        }
    }

class GeneratePromptReq(BaseModel):
    prompt: str = Field(description="Provided user prompt")
    model: str = Field(description="Model Name")

    # Pydantic v2 style
    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "Enhance this prompt."
            }
        }
    }