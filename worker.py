from multiprocessing import Queue
import logging as log
from .utils import *
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
    # output = llm.create_completion(pq, max_tokens=max_tokens, stop=stop)["choices"][0]["text"]
    gen = llm.generate(pq_tokens, top_k=40, top_p=0.95, temp=0.8, repeat_penalty=1.1)
    completion_tokens = []
    multibyte_fix = 0
    output = ""
    for (i, enc_tok) in enumerate(gen):
        if i >= max_tokens or enc_tok == llama_cpp.llama_token_eos():
            return llm.detokenize(completion_tokens[:max_tokens]).decode()
        completion_tokens.append(enc_tok)


        all_text = llm.detokenize(completion_tokens)

        # Stop incomplete bytes from passing, stolen from llama_cpp/llama.py
        for k, char in enumerate(all_text[-3:]):
            k = 3 - k
            for num, pattern in [(2, 192), (3, 224), (4, 240)]:
                # Bitwise AND check
                if num > k and pattern & char == pattern:
                    multibyte_fix = num - k

        if multibyte_fix > 0:
            multibyte_fix -= 1
            continue 

        all_text = all_text.decode()
        # stop processing code stolen from llama_cpp/llama.py
        any_stop = [s for s in stop if s in all_text]
        if len(any_stop) > 0:
            first_stop = any_stop[0]
            output = all_text[: all_text.index(first_stop)]
            break
    return output

def launch_worker(wid: int, prompt, state, iqueue, odict, cfg):
    log.info(f"Loading model for worker {wid}")
    llm = make_model(cfg) 
    while True:
        qid, query = iqueue.get()
        log.info(f"worker{wid} got query {qid}")
        start = time.time()
        output = generate_until(prompt,
                                query,
                                llm,
                                state,
                                max_tokens=cfg.max_tokens,
                                stop=cfg.stop)
        log.info(f"worker{wid} completed {qid} after {time.time() - start} seconds")
        odict[qid] = output
