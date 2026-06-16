import gradio as gr
from openai import OpenAI
import base64
import sqlite3
import time

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
DB_PATH = "/home/yavar/Documents/llm/metrics.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            latency REAL,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            tokens_per_sec REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

def log_metrics(latency, prompt_tokens, completion_tokens, tokens_per_sec):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO metrics (timestamp, latency, prompt_tokens, completion_tokens, tokens_per_sec) VALUES (?, ?, ?, ?, ?)",
        (time.time(), latency, prompt_tokens, completion_tokens, tokens_per_sec)
    )
    conn.commit()
    conn.close()


def clean_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return [c for c in content if not (isinstance(c, dict) and c.get("type") == "file")]
    if isinstance(content, dict) and content.get("type") == "file":
        return None
    return content

# Chat interface setup
def chat(message, history):
    system_prompt = {"role": "system", "content": "Do not use emojis or em dashes. Be a fair assistant. Keep responses short and concise. If user gives you images and is asking technical questions try to focuse and reason through"}

    content = []
    if message["text"]:
        content.append({"type": "text", "text": message["text"]})

    for file_path in message["files"]:
        with open(file_path, "rb") as f:
            b64_image = base64.b64encode(f.read()).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
        })

    clean_history = []
    for m in history:
        c = clean_content(m["content"])
        if c:
            clean_history.append({"role": m["role"], "content": c})

    messages = [system_prompt] + clean_history + [{"role": "user", "content": content}]

    start = time.time()
    response = client.chat.completions.create(
        model="Qwen/Qwen3-VL-32B-Instruct-FP8",
        messages=messages
    )
    latency = time.time() - start

    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    tokens_per_sec = completion_tokens / latency if latency > 0 else 0

    log_metrics(latency, prompt_tokens, completion_tokens, tokens_per_sec)
    print("METRICS LOGGED")

    return response.choices[0].message.content

demo = gr.ChatInterface(
    fn=chat,
    multimodal=True,
)
demo.launch(server_name="0.0.0.0")