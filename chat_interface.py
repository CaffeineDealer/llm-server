import gradio as gr
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Chat interface setup

def chat(message, history, file=None):

    system_prompt = {"role": "system", "content": "Do not use emojis or em dashes. Be a fair assistant. Keep responses short and concsie."}
    
    """Prepare history and add the Gradio interface"""
    messages = [system_prompt] + history + [{"role": "user", "content": message}]

    # if file is not None:
    #     # Convert image to base64 if needed (depending on model)
    #     if isinstance(file, str):  # if file is a path
    #         with open(file, "rb") as f:
    #             image_data = base64.b64encode(f.read()).decode("utf-8")
    #     else:  # if file is PIL Image or numpy array
    #         import io
    #         buf = io.BytesIO()
    #         file.save(buf, format="PNG")
    #         image_data = base64.b64encode(buf.getvalue()).decode("utf-8")

    #     # Format for Qwen-VL: use base64 encoded image in content
    #     user_message = {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": message},
    #             {"type": "image", "image": image_data}
    #         ]
    #     }
    # else:
    #     user_message = {"role": "user", "content": message}
    
    # messages.append(user_message)


    """Send messages to vLLM and get responses"""
    response = client.chat.completions.create(
        model="Qwen/Qwen3-VL-32B-Instruct-FP8",
        messages=messages
    )
    return response.choices[0].message.content

demo = gr.ChatInterface(
    fn=chat,
    # multimodal=True,
)
demo.launch(server_name="0.0.0.0")



    # if file:
    #     # For vLLM pass the file path or URL
    #     if isinstance(file, tuple):
    #         file_path = file[0]
    #     else:
    #         file_path = file

    #     # with open("image.jpg", "rb") as image_file:
    #     #     encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    #     user_content = [
    #         {"type": "text", "text": message},
    #         {"type": "image_url", "image_url": {"url": file_path}}
    #     ]
    #     messages.append({"role": "user", "content": user_content})
    # else:
    #     messages.append({"role": "user", "content": message})