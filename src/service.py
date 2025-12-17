import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from typing import List, Optional
import io
import base64
from logging import getLogger
from .config import settings

logger = getLogger(__name__)

class ImageService:
    def __init__(self):
        self.pipeline = None
        
    def load_model(self):
        logger.info(f"Loading model {settings.MODEL_ID}...")
        try:
            # Load with bfloat16 as per recommendation for efficiency
            pipe = QwenImageEditPlusPipeline.from_pretrained(
                settings.MODEL_ID, 
                torch_dtype=torch.bfloat16 if settings.DEVICE_MAP != "cpu" else torch.float32
            )
            
            if settings.DEVICE_MAP != "cpu" and torch.cuda.is_available():
                pipe.to("cuda")
                
            self.pipeline = pipe
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def generate(
        self,
        prompt: str,
        input_images_b64: Optional[List[str]] = None,
        negative_prompt: str = "",
        guidance_scale: float = 7.5,
        true_cfg_scale: float = 4.0,
        num_inference_steps: int = 40,
        seed: Optional[int] = None
    ) -> Image.Image:
        
        if not self.pipeline:
            raise RuntimeError("Model not initialized.")
            
        generator = None
        if seed is not None:
            generator = torch.manual_seed(seed)
            
        pil_images = []
        if input_images_b64:
            for b64 in input_images_b64:
                # Handle data URI scheme if present
                if ',' in b64:
                    b64 = b64.split(',')[1]
                image_data = base64.b64decode(b64)
                pil_images.append(Image.open(io.BytesIO(image_data)).convert("RGB"))
        
        # Prepare inputs
        inputs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "true_cfg_scale": true_cfg_scale,
            "generator": generator,
            "num_images_per_prompt": 1, # OpenAI API 'n' parameter handled at higher level usually by running loop or batching, here we do 1 for simplicity of this wrapper method
        }
        
        if pil_images:
            inputs["image"] = pil_images

        # Run inference
        # Note: The Qwen pipeline call signature might vary slightly based on version, 
        # but based on docs `image` is passed in kwargs/args.
        # The docs say: pipeline(**inputs)
        
        with torch.inference_mode():
            output = self.pipeline(**inputs)
            
        return output.images[0]

service = ImageService()
