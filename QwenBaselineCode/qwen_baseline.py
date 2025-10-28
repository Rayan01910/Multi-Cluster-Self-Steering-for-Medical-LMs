import torch
import torch.nn.functional as F
from system_prompt import get_system_prompt
from get_tag_contents import get_answer, exact_match_score
from normalize_answer import normalize_answer
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval_metrics import get_brier, get_ece, get_auroc
from load_medqa import load_medqa
import csv



def benchmark_qwen(dataset, model, tokenizer):
    csv_path="results.csv"
    all_confidences = []
    all_correctness = []
    count = 0
    rows = []

    sys_prompt = get_system_prompt() #change the sys prompt too
    
    for example in dataset:
        count = count + 1
        question = example['question']  
        options = example['options'] 
        correct_ans = example['answer_idx']
        
        prompt = f"Here is the question: {question}. These are the options: {options}."
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            return_dict_in_generate=True,
            output_scores=True,
        )
        
        generated_ids = outputs.sequences[:, inputs.input_ids.size(1):]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        answer = normalize_answer(get_answer(response))
        correct = exact_match_score(answer, correct_ans)
        
        logits = torch.stack(outputs.scores, dim=1) 
        logits = logits.to("cpu")
        
        confidences = F.softmax(logits, dim=-1)
        token_confidences = torch.max(confidences, dim=-1).values  
        softmax_confidence = token_confidences.mean().item()


        rows.append({
            "question": question,
            "answer_idx": correct_ans,
            "ai_answer": answer,
            "correct": correct,
            "confidence": softmax_confidence
        })
        
        print("Response:", response, flush=True)
        print("Correct Answer:", correct_ans, flush=True)
        print("AI Answer:", answer, flush=True)
        print("Correct:", correct, flush=True)
        
        print(f"Softmax-based confidence score: {softmax_confidence:.4f}", flush=True)
        
        all_confidences.append(softmax_confidence)
        all_correctness.append(correct)
        print(count, flush=True)
        
        # After all generation...

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer_idx", "ai_answer", "correct", "confidence"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    mean_auroc = get_auroc(all_correctness,all_confidences)
    mean_ece = get_ece(all_correctness,all_confidences)
    mean_brier = get_brier(all_correctness,all_confidences)

    print(f"AUROC: {mean_auroc}", flush=True)
    print(f"ECE: {mean_ece}", flush=True)
    print(f"Brier: {mean_brier}", flush=True)
    
    
    
