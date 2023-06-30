import os
from multiprocessing import Process, Queue, Manager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging as log
from .utils import *
from .worker import *


# check value of env var USE_GPU
use_gpu = os.environ.get("USE_GPU", False)
# CONFIG
root = "/home/garbus/interactivediffusion/zooweellama"
model = f"{root}/models/ggml-vic13b-q5_1.bin"
prompt_file = f"{root}/chat-with-vicuna-crossover.txt"
stop = ["Human:", "\n"]
max_tokens = 77
n_threads = 4 if use_gpu == "1" else 16
n_gpu_layers = 43 if use_gpu == "1" else 0
print(f"Using {n_gpu_layers} GPU layers")

class Prompt(BaseModel):
    prompt: str

with open(prompt_file, "r") as f:
    system_prompt = f.read()

llm = Llama(model_path=model, n_threads=n_threads, n_gpu_layers=n_gpu_layers)
prompt_tokens = llm.tokenize(system_prompt.encode())
llm.eval(prompt_tokens)
state = llm.save_state()
app = FastAPI()
@app.post("/generate")
def submit(p: Prompt):
    output = generate_until(system_prompt.strip(),
                            p.prompt.strip(),
                            llm,
                            state,
                            max_tokens=max_tokens,
                            stop=stop)
    print("-------------------------")
    print(f"{p.prompt}{output}")
    return output
