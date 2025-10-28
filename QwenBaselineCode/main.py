from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_baseline import benchmark_qwen 
from load_medqa import load_medqa 
import torch
import sys
import contextlib
from recalculate_metrics import recalculate_metrics

def benchmark_qwen_with_logging(dataset, model, tokenizer, log_file_path):
    with open(log_file_path, "w") as log_file, contextlib.redirect_stdout(log_file):
        # now all prints inside this block go to log_file
        benchmark_qwen(dataset, model, tokenizer)  # call your existing function here

    
    
def main():
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    log_file_path = "benchmark.log"
    #benchmark_qwen_with_logging(load_medqa("test"), model, tokenizer, log_file_path)
    recalculate_metrics()






if __name__ == "__main__":
    main()