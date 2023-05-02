import os
import sys
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager
from socketserver import ThreadingTCPServer,StreamRequestHandler
from threading import Lock
import logging as log
from utils import *
from worker import *
from config import *
setup_logging()

#cfg = benchmark1_config
# cfg = benchmark2_config
# cfg = benchmark3_config
cfg = benchmark4_config

processes = []

def setup(iqueue, odict):
    log.info(f"Loading prompt...")
    with open(cfg.prompt_file, "r") as f:
        prompt = f.read()
    log.info(f"Evaluating initial prompt")
    llm = make_model(cfg)
    prompt_tokens = llm.tokenize(prompt.encode())
    llm.eval(prompt_tokens)
    state = llm.save_state()
    log.info(f"Launching {cfg.num_llms} llm workers")
    for i in range(cfg.num_llms):
        p = Process(target=launch_worker,
                    args=[i, prompt, state, iqueue, odict, cfg])
        p.start()
        processes.append(p)

def start_webserver(iqueue, odict, qid, qlock, port=8000):
    class QueryHandler(StreamRequestHandler):
        def handle(self):
            log.info(f'Connected: {self.client_address[0]}:{self.client_address[1]}')
            while True:
                # get message
                msg = self.request.recv(4096).strip()#rfile.readline()
                if not msg:
                    log.info(f'Disconnected: {self.client_address[0]}:{self.client_address[1]}')
                    break # exits handler, framework closes socket
                # lock qid.value, increment it, then unlock
                qlock.acquire()
                qid.value += 1
                v = qid.value
                qlock.release()
                output = submit_query(v, msg.decode(), iqueue, odict)
                print(f'Received: {output}')
                self.wfile.write(output.encode())
                self.wfile.flush()

    log.info(f"Starting server on port {port}")
    server = ThreadingTCPServer(('',port),QueryHandler)
    server.allow_reuse_address = True
    server.serve_forever()


if __name__ == "__main__":
    log.info(f"---------STARTING-MAIN-----------")
    port = int(sys.argv[1])
    iqueue = Queue()
    qlock = Lock()
    with Manager() as manager:
        odict = manager.dict()
        qid = manager.Value('i', 0)
        setup(iqueue, odict)
        start_webserver(iqueue, odict, qid, qlock, port=port)
