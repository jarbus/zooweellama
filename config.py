class Config:
    def __init__(self, num_llms: int, threads_per_llm: int,
                max_tokens: int,
                model: str="../llama.cpp/models/ggml-vicuna-13b-1.1-q4_2.bin",
                prompt_file: str="chat-with-vicuna-crossover.txt",
                stop: list[str]=["Human:", "\n"]):

        self.num_llms = num_llms
        self.threads_per_llm = threads_per_llm
        self.model = model
        self.prompt_file = prompt_file
        self.stop = stop
        self.max_tokens = max_tokens


benchmark1_config = Config(num_llms=1,
                     threads_per_llm=32,
                     max_tokens=16)

benchmark2_config = Config(num_llms=2,
                     threads_per_llm=16,
                     max_tokens=16)

benchmark3_config = Config(num_llms=4,
                     threads_per_llm=8,
                     max_tokens=16)

benchmark4_config = Config(num_llms=8,
                     threads_per_llm=4,
                     max_tokens=16)
