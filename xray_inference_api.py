from model import generate_xray_from_text
from fastapi import FastAPI, HTTPException 
from fastapi.concurrency import run_in_threadpool
from Schemas import XrayRequest, XrayResponse
import base64
from io import BytesIO
from fastapi.responses import StreamingResponse
from PIL import Image
import os


app = FastAPI(
    title="X-ray Image Generation API",
    description="API to generate X-ray images from text prompts using a fine-tuned Stable Diffusion model.",
    version="1.0.0"
)



@app.post("/generate-xray", response_model=XrayResponse)
async def generate_xray(request: XrayRequest):
    """
    Generate an X-ray image from a text prompt.

    Args:
        request (XrayRequest): Request body with prompt and generation parameters

    Returns:
        XrayResponse: Base64-encoded image and save message
    """
    try:
        # Generate the image
        image = await run_in_threadpool(lambda: generate_xray_from_text(
            prompt=request.prompt,
            # model_path=request.model_path,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            negative_prompt=request.negative_prompt,
            output_file=request.output_file
        )
        )
        # Convert image to base64

        img_buffer = BytesIO()
        image.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        # buffered = BytesIO()
        # image.save(buffered, format="PNG")
        # image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Create save message
        save_message = f"Image saved to {request.output_file}" if request.output_file else "Image not saved (no output file specified)"

        # return XrayResponse(
        #     image_base64=image_base64,
        #     save_message=save_message
        # )
        return StreamingResponse(
            img_buffer,
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=xray.png"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)