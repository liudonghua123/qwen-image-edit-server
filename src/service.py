import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from typing import List, Optional
import io
import base64
from logging import getLogger
from .config import settings

logger = getLogger(__name__)

import asyncio

class ImageService:
    def __init__(self):
        self.pipeline = None
        self.queue = asyncio.Queue()

    async def start_worker(self):
        """Background worker to process requests from the queue."""
        logger.info("Starting model worker...")
        while True:
            params, future = await self.queue.get()
            try:
                # Run generation in a separate thread to avoid blocking the event loop
                # self.generate is CPU blocking (even with GPU, the python side waits)
                # Using asyncio.to_thread (available in Python 3.9+)
                result = await asyncio.to_thread(self.generate, **params)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            finally:
                self.queue.task_done()

    async def process_request(self, **kwargs):
        """Enqueue a request and wait for the result."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.queue.put((kwargs, future))
        return await future
        
    def load_model(self):
        logger.info(f"Loading model {settings.PRETRAINED_MODEL_NAME}...")
        try:
            # Load with bfloat16 as per recommendation for efficiency
            torch_dtype = torch.bfloat16 if settings.DEVICE_MAP != "cpu" else torch.float32
            
            kwargs = {}
            if settings.DEVICE_MAP != "cpu":
                kwargs["device_map"] = settings.DEVICE_MAP

            pipe = QwenImageEditPlusPipeline.from_pretrained(
                settings.PRETRAINED_MODEL_NAME, 
                torch_dtype=torch_dtype,
                **kwargs
            )

            self.pipeline = pipe
            logger.info(f"Model loaded successfully with device_map={settings.DEVICE_MAP}.")
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
