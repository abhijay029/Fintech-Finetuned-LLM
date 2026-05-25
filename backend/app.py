from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
AutoTokenizer,
AutoModelForCausalLM
)
import torch
from fastapi.middleware.cors import CORSMiddleware

model_path = "model/fine_tuned_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

app = FastAPI()

origins = [
    "http://localhost:5173",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_text(data: PromptRequest):
    
    inputs = tokenizer(
    data.prompt,
    return_tensors="pt"
    ).to(model.device)

    print("Inputs: ",inputs)

    outputs = model.generate(
    **inputs,
    max_new_tokens = 100
    )

    print("Output:",outputs)

    response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
    )

    print("Response: ",response)

    return {
    "response": response
    }