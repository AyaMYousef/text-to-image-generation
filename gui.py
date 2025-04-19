import gradio as gr
from model import generate_xray_from_text


def gradio_generate_xray(prompt, negative_prompt, height, width, num_inference_steps, guidance_scale, model_path, output_file):
    """
    Wrapper function for Gradio to call generate_xray_from_text and return the image and save message.
    """
    # Ensure height and width are multiples of 8 (required by Stable Diffusion)
    height = int(height // 8 * 8)
    width = int(width // 8 * 8)
    
    # Call the inference function
    image = generate_xray_from_text(
        prompt=prompt,
        model_path=model_path,
        height=height,
        width=width,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt if negative_prompt else None,
        output_file=output_file if output_file else None
    )
    
    # Return the image and a message
    save_message = f"Image saved to {output_file}" if output_file else "Image not saved (no output file specified)"
    return image, save_message

# Create Gradio interface
iface = gr.Interface(
    fn=gradio_generate_xray,
    inputs=[
        gr.Textbox(label="Prompt", value="An X-ray of a human chest", placeholder="Enter text description for the X-ray image"),
        gr.Textbox(label="Negative Prompt (optional)", placeholder="Enter negative prompt (e.g., blurry, low quality)"),
        gr.Slider(label="Height", minimum=256, maximum=1024, step=8, value=512),
        gr.Slider(label="Width", minimum=256, maximum=1024, step=8, value=512),
        gr.Slider(label="Number of Inference Steps", minimum=10, maximum=100, step=1, value=50),
        gr.Slider(label="Guidance Scale", minimum=1.0, maximum=20.0, step=0.5, value=7.5),
        gr.Textbox(label="Model Path", value="Stable-diffusion-finetuned/checkpoint-2200", placeholder="Path to model checkpoint"),
        gr.Textbox(label="Output File (optional)", placeholder="Enter path to save image (e.g., generated_xray.png)")
    ],
    outputs=[
        gr.Image(label="Generated X-ray Image"),
        gr.Text(label="Save Status")
    ],
    title="X-ray Image Generator",
    description="Generate X-ray images from text prompts using a fine-tuned Stable Diffusion model.",
    theme="default"
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()