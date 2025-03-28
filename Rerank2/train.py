
import torch
import torch.nn as nn
import argparse
from retriever import deep_cluster, k_means_cluster, bm25_select_profiles
import json
import yaml
from model import BertForClassification
from transformers import BertTokenizer, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from copy import deepcopy
from retriever import build_corpus_and_query
from prompts.prompts import simple_prompt_generator
from datasets import Dataset  # 导入HF Dataset
from tqdm.auto import tqdm

def generate_data(model, tokenizer, source_data, cluster_data, target_data, task, task_config, k):
    # 1. 预处理优化
    max_length = 4096
    device = model.device
    prompt_generator = simple_prompt_generator(max_length, tokenizer)
    
    # 2. 批量检索（假设bm25_select_profiles支持批量）
    retrieval_data = bm25_select_profiles(cluster_data, task, task_config, k)
    
    # 3. 预生成所有prompt变体（添加进度条）
    batch_inputs = []
    batch_targets = []
    inputs = []
    
    # 计算总进度步数
    total_profiles = sum(len(data['profile']) for data in source_data)
    total_combinations = total_profiles * k
    
    with tqdm(total=total_combinations, 
             desc="🔄 Generating prompt variants",
             unit="combo") as pbar:
        
        for data_idx, data in enumerate(source_data):
            profiles = data['profile']
            current_retrieval = retrieval_data[data_idx]
            
            # 预生成基础语料
            corpus, _, _ = build_corpus_and_query(task, data['input'], profiles, task_config)
            inputs.append(corpus)
            
            # 生成所有可能的profile替换组合
            for profile_idx, profile in enumerate(profiles):
                for pos in range(k):
                    modified_retrieval = current_retrieval.copy()
                    modified_retrieval["profile"][pos] = profile
                    prompt = prompt_generator(data['input'], modified_retrieval["profile"], task)
                    full_text = prompt + target_data[data_idx]["output"]
                    batch_inputs.append(full_text)
                    batch_targets.append(len(prompt))
                    
                    # 更新进度条
                    pbar.update(1)
                    pbar.set_postfix({
                        'data_idx': data_idx,
                        'current_profile': f"{profile_idx+1}/{len(profiles)}",
                        'current_pos': pos+1
                    })

    # 4. 批量编码（添加子进度条）
    with tqdm(total=1, desc="📥 Batch encoding", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        encoded_batch = tokenizer(
            batch_inputs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=True
        ).to(device)
        pbar.update(1)

    # 5. 批量前向传播（添加子进度条）
    with tqdm(total=1, desc="🤖 Model forwarding", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        with torch.no_grad():
            outputs = model(**encoded_batch, labels=encoded_batch.input_ids)
        pbar.update(1)
    
    # 6. 并行计算所有样本的log_prob（添加进度条）
    log_probs = []
    with tqdm(total=len(batch_inputs), 
             desc="🧮 Calculating log probs",
             unit="sample") as pbar:
        
        for idx in range(len(batch_inputs)):
            prompt_len = batch_targets[idx]
            target_ids = encoded_batch.input_ids[idx, prompt_len:]
            
            # 获取对应位置的logits
            logits = outputs.logits[idx, prompt_len-1:-1, :]
            
            # 向量化计算
            log_prob = torch.log_softmax(logits, dim=-1)
            token_log_probs = log_prob.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            log_probs.append(token_log_probs.sum().item())
            
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({
                'current_logprob': f"{log_probs[-1]:.2f}",
                'avg_logprob': f"{sum(log_probs)/len(log_probs):.2f}"
            })

    # 7. 重组结果（添加进度条）
    labels = []
    ptr = 0
    
    with tqdm(total=len(source_data), 
             desc="📦 Packaging results",
             unit="data") as pbar:
        
        for data_idx, data in enumerate(source_data):
            for _ in data['profile']:
                labels.append(log_probs[ptr:ptr+k])
                ptr += k
            pbar.update(1)
            pbar.set_postfix({
                'current_data': data_idx+1,
                'profiles_packed': len(data['profile'])
            })
    
    return inputs, labels

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)

if __name__ == "__main__":
    opts = parser.parse_args()
    task = opts.task

    source_data_addr = f"data/{task}/train_questions.json"
    target_data_addr = f"data/{task}/train_outputs.json"

    with open(source_data_addr) as file:
        source_data = json.load(file)
    with open(target_data_addr) as file:
        target_data = json.load(file)["golds"]
    
    config_path = "configs/base.yaml"
    config = yaml.safe_load(open(config_path, 'r'))
    task_config = config["task_configurations"].get(task)

    k = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classification_model_name = "bert-base-uncased"
    classification_model = BertForClassification(classification_model_name, num_labels=k).to(device)
    classification_tokenizer = BertTokenizer.from_pretrained(classification_model_name)

    generation_model_name = "gpt2"
    generation_model = AutoModelForCausalLM.from_pretrained(generation_model_name).to(device)
    generation_tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
    
    cluster_data = k_means_cluster(source_data, task, task_config, k)

    inputs, labels = generate_data(generation_model, generation_tokenizer, source_data, cluster_data, target_data, task, task_config, k)
    
    # 创建HF Dataset
    hf_dataset = Dataset.from_dict({
        "text": inputs,
        "labels": labels
    })
    
    # 定义tokenize函数
    def tokenize_function(examples):
        return classification_tokenizer(
            examples["text"],
            padding="max_length",  # 确保填充到固定长度
            truncation=True,
            max_length=512,
            return_tensors=None  # 返回Python列表而非张量
        )
    
    # 应用tokenizer
    tokenized_dataset = hf_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]  # 移除原始文本列
    )
    
    # 将数据集格式设置为PyTorch张量
    tokenized_dataset.set_format(
        type="torch", 
        columns=["input_ids", "attention_mask", "labels"],
        output_all_columns=False
    )
    
    # 现在可以直接使用tokenized_dataset进行训练
    # 例如：传递给Trainer或自定义DataLoader

    trainning_args = TrainingArguments(
        output_dir="output",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        logging_dir="logs",
        logging_steps=50,
        save_strategy="no",
        learning_rate=5e-5,
        evaluation_strategy="no",
        report_to="none",
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=classification_model,
        args=trainning_args,
        train_dataset=tokenized_dataset
    )

    trainer.train()