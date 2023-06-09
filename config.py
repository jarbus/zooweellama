root="/home/garbus/genetic_prompting/llambalancer"
class Config:
    def __init__(self, num_llms: int, threads_per_llm: int,
                #model: str="../llama.cpp/models/ggml-vicuna-13b-1.1-q4_2.bin",
                model: str=f"{root}/models/ggml-vic13b-q5_1.bin",
                prompt_file: str=f"{root}/chat-with-vicuna-crossover.txt",
                stop: list[str]=["Human:", "\n"],
                max_tokens: int=128):

        self.num_llms = num_llms
        self.threads_per_llm = threads_per_llm
        self.model = model
        self.prompt_file = prompt_file
        self.stop = stop
        self.max_tokens = max_tokens


benchmark1_config = Config(num_llms=1,
                     threads_per_llm=32,)

benchmark2_config = Config(num_llms=2,
                     threads_per_llm=16,)

benchmark3_config = Config(num_llms=4,
                     threads_per_llm=12,)

benchmark4_config = Config(num_llms=8,
                     threads_per_llm=4)

# benchmark5_config = Config(num_llms=14,
#                      threads_per_llm=4,
#                      max_tokens=48)
