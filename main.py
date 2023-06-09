import os
import sys
from multiprocessing import Process, Queue, Manager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging as log
from .utils import *
from .worker import *


# CONFIG
root = "/home/garbus/genetic_prompting/llambalancer"
model = f"{root}/models/ggml-vic13b-q5_1.bin"
prompt_file = f"{root}/chat-with-vicuna-crossover.txt"
stop = ["Human:", "\n"]
max_tokens = 128
n_threads = 8

class Prompt(BaseModel):
    prompt: str

with open(prompt_file, "r") as f:
    system_prompt = f.read()

llm = Llama(model_path=model, n_threads=n_threads)
prompt_tokens = llm.tokenize(system_prompt.encode())
llm.eval(prompt_tokens)
state = llm.save_state()
app = FastAPI()
@app.post("/")
def submit(p: Prompt):
    global iqueue, odict, qlock, qid
    print(f"Received prompt: {p.prompt}")
    output = generate_until(system_prompt,
                            p.prompt,
                            llm,
                            state,
                            max_tokens=max_tokens,
                            stop=stop)
    print("Output:", output)
    return output
