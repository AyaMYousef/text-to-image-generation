from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class XrayRequest(BaseModel):
    """Model Request Schema"""
    prompt: str
    # model_path: str = "D:/Projects/DEPI/T2I/Stable-diffusion-finetuned/checkpoint-2200"
    height: int = 512
    width: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    negative_prompt: Optional[str] = None
    output_file: Optional[str] = None


class XrayResponse(BaseModel):
    """Model Response Schema"""
    image_base64: str
    save_message: str