import time
import asyncio
import socket
import sys

HOST, PORT = "localhost", int(sys.argv[1])

async def submit_query(q):
    print("Creating task")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Connect to server and send data
        sock.connect((HOST, PORT))
        sock.sendall(bytes(q + "\n", "utf-8"))
        print("Sent data")

        await asyncio.sleep(1)
        # Receive data from the server and shut down
        received = str(sock.recv(1024), "utf-8")
        print(f"Received: {received}")
        return received

async def submit_n_queries(n):

    #data = " ".join(sys.argv[1:])
    data = """Human: Caption 1: a laptop, retro
    Caption 2: cyborg from teen titans, noir
    Assistant: """
    tasks = []
    for i in range(n):
        # make 4 tasks and wait for them to finish
        print(f"Creating task {i}")
        task = asyncio.create_task(submit_query(data))
        print(f"Done")
        tasks.append(task)
    results = await asyncio.gather(*tasks)


if __name__ == "__main__":
    start = time.time()
    asyncio.run(submit_n_queries(16))
    end = time.time()
    print(f"Time: {end-start}")
