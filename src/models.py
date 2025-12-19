from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal, Union
import base64
import io

class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description="A text description of the desired image(s).")
    n: int = Field(1, description="The number of images to generate. Must be between 1 and 10.")
    size: str = Field("1024x1024", description="The size of the generated images. Must be one of 256x256, 512x512, or 1024x1024.")
    response_format: Literal["url", "b64_json"] = Field("url", description="The format in which the generated images are returned.")
    user: Optional[str] = Field(None, description="A unique identifier representing your end-user.")
    
    # Extended fields for Qwen-Image-Edit-2509
    input_images: Optional[List[str]] = Field(None, description="List of base64 encoded input images.")
    negative_prompt: Optional[str] = Field(None, description="The prompt or prompts not to guide the image generation.")
    guidance_scale: Optional[float] = Field(None, description="Guidance scale as defined in the Diffusers library.")
    true_cfg_scale: Optional[float] = Field(None, description="True CFG scale.")
    num_inference_steps: Optional[int] = Field(None, description="The number of denoising steps.")
    seed: Optional[int] = Field(None, description="Random seed for generation.")

    @field_validator('input_images')
    @classmethod
    def validate_input_images(cls, v):
        if v:
            for img_str in v:
                if not isinstance(img_str, str):
                     raise ValueError("Input images must be base64 strings")
        return v

class ImageResult(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None

class ImageGenerationResponse(BaseModel):
    created: int
    data: List[ImageResult]

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 1677610602
    owned_by: str = "custom"

class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelCard]
