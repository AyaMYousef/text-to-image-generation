import gradio as gr
from model import generate_xray_from_text
from chatbot import get_agent_response
from typing import List, Tuple



def gradio_generate_xray(prompt, negative_prompt, height, width, num_inference_steps, guidance_scale, model_path, output_file):
    """
    Wrapper function for Gradio to call generate_xray_from_text and return the image and save message.
    """
    height = int(height // 8 * 8)
    width = int(width // 8 * 8)
    
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
    
    save_message = f"Image saved to {output_file}" if output_file else "Image not saved (no output file specified)"
    return image, save_message

def send_chat_message(chat_input: str, chatbot: List[Tuple[str, str]]) -> tuple[List[Tuple[str, str]], str]:
    """
    Function to handle the Send button click or Enter key for chatbot input.
    Takes the chat input, updates the chatbot history, and clears the input textbox.
    """
    if not chat_input:
        return chatbot, ""  # Return unchanged if input is empty
    
    # Get response from the agent
    response = get_agent_response(chat_input , user_id= "john_doe@example.com")
    
    # Append input and response to chatbot history
    updated_chatbot = chatbot + [(chat_input, response)]
    
    # Return updated chatbot and clear the input textbox
    return updated_chatbot, ""


def generate_description(prompt: str, chatbot: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Function to handle the Generate Description button click.
    Takes the prompt from the Prompt textbox and updates the chatbot.
    """
    if not prompt:
        return chatbot  # Return unchanged if prompt is empty
    
    # Get response from the agent
    response = get_agent_response(prompt  , user_id= "john_doe@example.com")
    
    # Update chatbot with user prompt and agent response
    updated_chatbot = chatbot + [(prompt, response)]
    return updated_chatbot

# Create Gradio interface with custom layout
with gr.Blocks(theme="default Protestant") as theme:
    gr.Markdown("# X-ray Image Generator")
    gr.Markdown("Generate X-ray images from text prompts using a fine-tuned Stable Diffusion model.")

    with gr.Row():
        with gr.Column():
            # Input components for X-ray generation
            prompt = gr.Textbox(
                label="Prompt",
                value="An X-ray of a human chest",
                placeholder="Enter text description for the X-ray image"
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt (optional)",
                placeholder="Enter negative prompt (e.g., blurry, low quality)"
            )
            height = gr.Slider(label="Height", minimum=256, maximum=1024, step=8, value=512)
            width = gr.Slider(label="Width", minimum=256, maximum=1024, step=8, value=512)
            num_inference_steps = gr.Slider(
                label="Number of Inference Steps",
                minimum=10,
                maximum=100,
                step=1,
                value=10
            )
            guidance_scale = gr.Slider(
                label="Guidance Scale",
                minimum=1.0,
                maximum=20.0,
                step=0.5,
                value=7.5
            )
            model_path = gr.Textbox(
                label="Model Path",
                value="Stable-diffusion-finetuned/checkpoint-2200",
                placeholder="Path to model checkpoint"
            )
            output_file = gr.Textbox(
                label="Output File (optional)",
                placeholder="Enter path to save image (e.g., generated_xray.png)"
            )
            
            # Buttons
            generate_image_btn = gr.Button("Generate X-ray Image")
            generate_desc_btn = gr.Button("Generate Description")

        with gr.Column():
            # Output components
            output_image = gr.Image(label="Generated X-ray Image")
            save_status = gr.Text(label="Save Status")
            chatbot = gr.Chatbot(label="Chat with Medical Guide")
            chat_input = gr.Textbox(
                label="Chat Input",
                placeholder="Type your question or prompt for the medical guide...",
                lines=2
            )
            send_chat_btn = gr.Button("Send")

    # Event handlers
    generate_image_btn.click(
        fn=gradio_generate_xray,
        inputs=[
            prompt,
            negative_prompt,
            height,
            width,
            num_inference_steps,
            guidance_scale,
            model_path,
            output_file
        ],
        outputs=[output_image, save_status]
    )

    generate_desc_btn.click(
        fn=generate_description,
        inputs=[prompt, chatbot],
        outputs=[chatbot]
    )
    chat_input.submit(
        fn=generate_description,
        inputs=[chat_input, chatbot],
        outputs=[chatbot, chat_input]
    )
    send_chat_btn.click(
        fn=send_chat_message,
        inputs=[chat_input, chatbot],
        outputs=[chatbot, chat_input]
    )

# Launch the interface
if __name__ == "__main__":
    theme.launch()