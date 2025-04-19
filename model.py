from diffusers import StableDiffusionPipeline
import torch
from fastapi import HTTPException


# Function to generate X-ray images from text
def generate_xray_from_text(
    prompt,
    model_path="Stable-diffusion-finetuned/checkpoint-2200", # change this ?
    height=512,
    width=512,
    num_inference_steps=50,
    guidance_scale=7.5,
    negative_prompt=None,
    output_file=None
):
    """
    Generate an X-ray image from a text prompt using a fine-tuned Stable Diffusion model.

    Args:
        prompt (str): Text description of the X-ray image to generate
        model_path (str): Path to the fine-tuned model checkpoint
        height (int): Height of output image
        width (int): Width of output image
        num_inference_steps (int): Number of denoising steps
        guidance_scale (float): Scale for classifier-free guidance
        negative_prompt (str, optional): Negative prompt to guide generation
        output_file (str, optional): Path to save the generated image

    Returns:
        PIL.Image: Generated X-ray image
    """
    # Load the fine-tuned pipeline
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = pipeline.to(device)

        # Generate the image
        with torch.no_grad():
            image = pipeline(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt
            ).images[0]

        # Save the image if output path is provided
        if output_file:
        
            image.save(output_file)
            print(f"Image saved to {output_file}")

        return image
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate image: {str(e)}")


# prompt = "Chest X-ray showing pneumonia in the right lower lobe"
# generate_xray_from_text(prompt, output_file="generated_xray.png")