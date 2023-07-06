import os
from copy import deepcopy
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
# model = f"{root}/models/ggml-vic13b-q5_1.bin"
model = f"{root}/models/vicuna-7b-v1.3.ggmlv3.q5_1.bin"
stop = ["Human:", "\n"]
max_tokens = 77
n_threads = 4 if use_gpu == "1" else 24
n_gpu_layers = 43 if use_gpu == "1" else 0
print(f"Using {n_gpu_layers} GPU layers")

class Prompt(BaseModel):
    prompt: str

#prompt_file = f"{root}/chat-with-vicuna-crossover.txt"
extract_prompt_file = f"{root}/chat-with-vicuna-extract.txt"
combine_prompt_file = f"{root}/chat-with-vicuna-combine.txt"
crossover_prompt_file = f"{root}/chat-with-vicuna-crossover.txt"
sub_extract_prompt_file = f"{root}/chat-with-vicuna-subject-extract.txt"
sub_reinsert_prompt_file = f"{root}/chat-with-vicuna-subject-reinsert.txt"

with open(extract_prompt_file, "r") as f:
    extract_prompt = f.read().strip()
with open(combine_prompt_file, "r") as f:
    combine_prompt = f.read().strip()
with open(crossover_prompt_file, "r") as f:
    crossover_prompt = f.read().strip()
with open(sub_extract_prompt_file, "r") as f:
    sub_extract_prompt = f.read().strip()
with open(sub_reinsert_prompt_file, "r") as f:
    sub_reinsert_prompt = f.read().strip()

llm = Llama(model_path=model, n_threads=n_threads, n_gpu_layers=n_gpu_layers)
# eval extract
prompt_tokens = llm.tokenize(extract_prompt.encode())
llm.eval(prompt_tokens)
extract_state = llm.save_state()
# eval combine
prompt_tokens = llm.tokenize(combine_prompt.encode())
llm.eval(prompt_tokens)
combine_state = llm.save_state()
# eval crossover
prompt_tokens = llm.tokenize(crossover_prompt.encode())
llm.eval(prompt_tokens)
crossover_state = llm.save_state()


app = FastAPI()
@app.post("/extract")
def extract(p: Prompt):
    ex_desc = generate_until(extract_prompt,
                            p.prompt.strip(),
                            llm,
                            extract_state,
                            max_tokens=max_tokens,
                            stop=stop)

    print("-------------------------")
    print(f"{p.prompt}\nEXTRACT: {ex_desc}")
    return ex_desc

@app.post("/combine")
def combine(p: Prompt):
    comb_prompt = generate_until(combine_prompt,
                            p.prompt.strip(),
                            llm,
                            combine_state,
                            max_tokens=max_tokens,
                            stop=stop)
    print(f"{p.prompt}\nCOMBINE: {comb_prompt}")
    return comb_prompt


@app.post("/crossover")
def crossover(p: Prompt):
    crossed_prompt = generate_until(crossover_prompt,
                            p.prompt.strip(),
                            llm,
                            crossover_state,
                            max_tokens=max_tokens,
                            stop=stop)
    print(f"{p.prompt}\CROSSED: {crossed_prompt}")
    return crossed_prompt

@app.post("/subject-extract")
def subject_extract(p: Prompt):
    subex = generate_until(sub_extract_prompt,
                            p.prompt.strip(),
                            llm,
                            None,
                            max_tokens=max_tokens,
                            stop=stop)

    print(f"{p.prompt}\nSUBEXTRACT: {subex}")
    return subex

@app.post("/subject-reinsert")
def subject_reinsert(p: Prompt):
    subre = generate_until(sub_reinsert_prompt,
                            p.prompt.strip(),
                            llm,
                            None,
                            max_tokens=max_tokens,
                            stop=stop)

    print(f"{p.prompt}\nSUBREINSERT: {subre}")
    return subre
