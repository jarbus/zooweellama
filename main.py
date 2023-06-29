import sys
import traceback
from multiprocessing import Process, Queue, Manager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .worker import *
import json
from asyncio import Lock


# CONFIG
root = "/home/garbus/interactivediffusion/zooweellama"
model = f"{root}/models/ggml-vic13b-q5_1.bin"
json_cot_prompt_file = f"{root}/chat-with-vicuna-crossover.txt"
original_prompt_file = f"{root}/chat-with-vicuna-crossover.txt-original"
# prompt_file = f"{root}/chat-with-vicuna-shuffle-phrases.txt"
stop = ["Human:", "\n"]
max_tokens = 9999
n_threads = 8
n_gpu_layers = 43

class Prompt(BaseModel):
    prompt: str

with open(json_cot_prompt_file, "r") as f:
    json_cot_system_prompt = f.read()
with open(original_prompt_file, "r") as f:
    original_system_prompt = f.read()

llm = Llama(model_path=model, n_threads=n_threads, n_gpu_layers=n_gpu_layers)

llm.eval(llm.tokenize(json_cot_system_prompt.encode()))
json_cot_state = llm.save_state()

app = FastAPI()
@app.post("/generate")
async def submit(p: Prompt):
    global json_cot_state

    json_cot_output = generate_until(json_cot_system_prompt,
                            p.prompt,
                            llm,
                            json_cot_state,
                            max_tokens=max_tokens,
                            stop=stop)
    print("-----------------------------")
    print(p.prompt.strip(), end="")
    try:
        json_cot_output = json_cot_output.replace("\_","_")
        json_cot_output = json.loads(json_cot_output)
        return json_cot_output["combined_caption"]
    except Exception as e:
        print("ERROR, DEFAULTING TO ORIGINAL", file=sys.stderr)
        print(json_cot_output, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        original_output = generate_until(original_system_prompt,
                                    p.prompt,
                                    llm,
                                    None,
                                    max_tokens=max_tokens,
                                    stop=stop)
        llm.eval(llm.tokenize(json_cot_system_prompt.encode()))
        json_cot_state = llm.save_state()
        return original_output

