from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# Define the input model
class PromptRequest(BaseModel):
    prompt: str

# Define the guardrail class
class guardrail:
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained("ProtectAI/deberta-v3-base-prompt-injection")
        model = AutoModelForSequenceClassification.from_pretrained("ProtectAI/deberta-v3-base-prompt-injection")

        self.classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            truncation=True,
            max_length=512,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    def guard(self, prompt):
        return self.classifier(prompt)

# Initialize FastAPI app
app = FastAPI()

# Initialize guardrail instance
guardrail_instance = guardrail()

# Define the FastAPI endpoint
@app.post("/guard/")
async def guard_prompt(request: PromptRequest):
    try:
        result = guardrail_instance.guard(request.prompt)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


