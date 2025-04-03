from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForCausalLM
# from transformers.models.llama import LlamaTokenizer
from transformers.data.data_collator import DataCollatorForSeq2Seq
import argparse
from metrics.classification_metrics import create_metric_f1_accuracy, create_metric_mae_rmse
from metrics.generation_metrics import create_metric_bleu_rouge_meteor
from data.datasets import get_all_labels, GeneralSeq2SeqDataset, create_preprocessor, convert_to_hf_dataset, SimpleDataset
from prompts.prompts import create_prompt_generator
from prompts.prompts import simple_prompt_generator
import json
import os
import yaml
from retriever import random_select_profiles, bm25_select_profiles, k_means_cluster
import sys
from tqdm.auto import tqdm
from retriever import build_corpus_and_query
import torch

parser = argparse.ArgumentParser()

parser.add_argument("--task", required = True)
parser.add_argument("--experiment_name", required = True)
parser.add_argument("--max_length", type = int, default = 1024)
parser.add_argument("--generation_max_length", type = int, default = 128)
parser.add_argument("--per_device_batch_size", type = int, default = 1)
parser.add_argument("--generation_num_beams", type = int, default = 1)
parser.add_argument("--cache_dir", default = "./cache")



if __name__ == "__main__":

    opts = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Qwen/Qwen1.5-7B"
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=opts.cache_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=opts.cache_dir)
    #collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = model, max_length = opts.max_length)

    output_dir = "experiments/{}/{}".format(opts.experiment_name, opts.task)

    task = opts.task
    prompt_generator = simple_prompt_generator()
    
    source_data_addr = "data/{}/dev_questions.json".format(task)
    target_data_addr = "data/{}/dev_outputs.json".format(task)



    with open(source_data_addr) as file:
        source_data = json.load(file)
    
    with open(target_data_addr) as file:
        target_data = json.load(file)["golds"]

    
    

    config_path = "configs/base.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)  # 安全加载方式，推荐使用

    task_config = config["task_configurations"].get(task)
    if not task_config:
        raise ValueError(f"Unsupported task: {task}. Available tasks: {list(config['task_configurations'].keys())}")
    
    source_data = source_data[0:1]
    cluster_data = k_means_cluster(source_data, task, task_config, k = 1)
    retrieval_data = bm25_select_profiles(cluster_data, task, task_config, k = 1)


    prompt_generator = simple_prompt_generator()


    
    # 计算总进度步数
    total_users = len(retrieval_data)
    
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_predictions = []
    all_references = []

    with tqdm(total=total_users, 
             desc="Generating predictions") as pbar:
        
        for data_idx, data in enumerate(retrieval_data):
            prompt = prompt_generator(data['input'], data['profile'], task)
            target_output = target_data[data_idx]["output"]
            
            # 生成预测
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=opts.max_length).to(model.device)
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=opts.generation_max_length,
                num_beams=opts.generation_num_beams,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # 解码生成文本（跳过prompt部分）
            generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
            predicted_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            all_predictions.append(predicted_text)
            all_references.append(target_output)
            
            
            pbar.update(1)

    # 计算评估指标
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from rouge import Rouge 
    import numpy as np

    # BLEU计算（使用nltk）
    bleu_references = [[ref.split()] for ref in all_references]
    bleu_candidates = [pred.split() for pred in all_predictions]
    bleu_score = corpus_bleu(bleu_references, bleu_candidates, 
                            smoothing_function=SmoothingFunction().method4)

    # ROUGE计算（使用rouge库）
    rouge = Rouge()
    rouge_scores = rouge.get_scores(all_predictions, all_references, avg=True)
    
    print(f"\nEvaluation results for {opts.task}:")
    print(f"BLEU: {bleu_score:.4f}")
    print(f"ROUGE-1: {rouge_scores['rouge-1']['f']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rouge-l']['f']:.4f}")

    # 保存结果
    results = {
        "bleu": bleu_score,
        "rouge1": rouge_scores['rouge-1']['f'],
        "rougeL": rouge_scores['rouge-l']['f'],
        "predictions": all_predictions,
        "references": all_references
    }
    
    output_dir = f"experiments/{opts.experiment_name}/{opts.task}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {output_dir}/results.json")