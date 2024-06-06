import asyncio
import json
import time
import requests
from transformers import AutoTokenizer
from typing import Optional, List

from pydantic import BaseModel, Field

from starlette.responses import StreamingResponse
from fastapi import FastAPI, HTTPException, Request

# Initialize the tokenizer
checkpoint = "vicgalle/Unsafe-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
app = FastAPI(title="OpenAI-compatible API")

def cleaner(text):
    # Split the text using "assistant\n\n" as the delimiter
    parts = text.split("assistant<|end_header_id|>\n\n")

    # Check if there are at least two parts after splitting
    if len(parts) > 2:
        # Return the content after the second occurrence
        return parts[2].strip()
    else:
        return None

# data models
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "mock-gpt-model"
    messages: List[Message]
    max_tokens: Optional[int] = 768
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False

def get_completion(prompt):
    url = "http://34.32.181.198:8005/v1/completions"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "vicgalle/Unsafe-Llama-3-8B",
        "prompt": prompt,
        "max_tokens": 768,
        "temperature": 0
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))


    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "message": response.text}
async def _resp_async_generator(text_resp: str):
    # let's pretend every word is a token and return it over time
    tokens = text_resp.split(" ")

    for i, token in enumerate(tokens):
        chunk = {
            "id": i,
            "object": "chat.completion.chunk",
            "created": time.time(),
            "model": 'balle',
            "choices": [{"delta": {"content": token + " "}}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.05)
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if request.messages:
        prompt = tokenizer.apply_chat_template(request.messages, tokenize=False, add_generation_prompt=False)
        result = get_completion(prompt)
        if "choices" in result and len(result["choices"]) > 0:
            text = result["choices"][0]["text"]
            resp_content = cleaner(text)
        else:
            resp_content = "No valid response received from the assistant."

    else:
        resp_content = "As a mock AI Assitant, I can only echo your last message, but there wasn't one!"
    if request.stream:
        return StreamingResponse(
            _resp_async_generator(resp_content), media_type="application/x-ndjson"
        )
    return {
        "id": "1337",
        "object": "chat.completion",
        "created": time.time(),
        "model": request.model,
        "choices": [{"message": Message(role="assistant", content=resp_content)}],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
