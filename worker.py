from multiprocessing import Queue
import logging as log
from utils import *
from llama_cpp import Llama
import llama_cpp
import time

def make_model(cfg):
    return Llama(model_path=cfg.model, n_threads=cfg.threads_per_llm)

def submit_query(qid: int,
                 query: str,
                 iqueue: "Queue",
                 odict: "Manager.dict"):
    log.info(f"Submitted query {qid}")
    iqueue.put((qid, query))
    while qid not in odict:
        time.sleep(0.1)

    output = odict[qid]
    del odict[qid]
    return output

def generate_until(prompt: str, query: str, llm, state, stop: list[str], max_tokens: int=128):
    llm.load_state(state)
    pq = prompt + query
    pq_tokens = llm.tokenize(pq.encode())
    output: list[bytes] = []
    gen = llm.generate(pq_tokens, top_k=40, top_p=0.95, temp=0.8, repeat_penalty=1.1)
    for (i, enc_tok) in enumerate(gen):
        if enc_tok == llama_cpp.llama_token_eos():
            break
        dec_tok = llm.detokenize([enc_tok])
        output.append(dec_tok)
        if dec_tok in stop or i >= max_tokens:
            break
    output = (b"".join(output)).decode()
    return output

def launch_worker(wid: int, prompt, state, iqueue, odict, cfg):
    log.info(f"Loading model for worker {wid}")
    llm = make_model(cfg) 
    while True:
        qid, query = iqueue.get()
        log.info(f"worker{wid} got query {qid}")
        start = time.time()
        output = generate_until(prompt, query, llm, state, max_tokens=cfg.max_tokens, stop=cfg.stop)
        log.info(f"worker{wid} completed {qid} after {time.time() - start} seconds")
        odict[qid] = output
