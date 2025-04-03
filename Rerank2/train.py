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
import os

def generate_data(model, tokenizer, source_data, cluster_data, target_data, task, task_config, k):
    # 1. 预处理优化
    max_length = 1024
    device = model.device
    prompt_generator = simple_prompt_generator()
    
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
            
            # 生成所有可能的profile替换组合
            for profile_idx, profile in enumerate(profiles):
                inputs.append(corpus[profile_idx])
                for pos in range(k):
                    modified_retrieval = current_retrieval.copy()
                    modified_retrieval["profile"][pos] = profile
                    prompt = prompt_generator(data['input'], modified_retrieval["profile"], task)
                    target_output = target_data[data_idx]["output"]
                    
                    # ============== 关键修改开始 ==============
                    # 分别编码prompt和target（不添加特殊token）
                    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
                    target_tokens = tokenizer.encode(target_output, add_special_tokens=False)
                    
                    # 合并为完整token序列并解码
                    full_tokens = prompt_tokens + target_tokens
                    full_text = tokenizer.decode(full_tokens, skip_special_tokens=True)
                    
                    batch_inputs.append(full_text)
                    batch_targets.append(len(prompt_tokens))  # 存储token级长度
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'data_idx': data_idx,
                        'current_profile': f"{profile_idx+1}/{len(profiles)}",
                        'current_pos': pos+1
                    })

    log_probs = []
    
    # 设置子批次大小以减少显存消耗
    sub_batch_size = 4 # 根据显存情况调整该值
    
    # 主进度条（包含子批次处理）
    with tqdm(total=len(batch_inputs), desc="🔁 Processing sub-batches", 
             unit="sample", postfix={"sub_batch": 0}) as main_pbar:
        
        # 分批次处理数据
        for sub_idx in range(0, len(batch_inputs), sub_batch_size):
            sub_inputs = batch_inputs[sub_idx:sub_idx + sub_batch_size]
            sub_targets = batch_targets[sub_idx:sub_idx + sub_batch_size]
            
            # 更新主进度条状态
            main_pbar.set_postfix({"sub_batch": sub_idx//sub_batch_size + 1})
            

            
    
            # 4. 子批次编码（无内部进度条）
            encoded_sub = tokenizer(
                sub_inputs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                add_special_tokens=True
            ).to(device)

            
            # 5. 子批次前向传播（无内部进度条）
            with torch.no_grad():
                outputs = model(**encoded_sub, labels=encoded_sub.input_ids)
                
                    
            # 6. 计算子批次log_probs（无内部进度条）
            for idx in range(len(sub_inputs)):
                prompt_len = sub_targets[idx]
                target_ids = encoded_sub.input_ids[idx, prompt_len:]
                logits = outputs.logits[idx, prompt_len-1:-1, :]
                log_prob = torch.log_softmax(logits, dim=-1)
                token_log_probs = log_prob.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
                log_probs.append(token_log_probs.sum().item())
                
            
            # 更新主进度条
            main_pbar.update(len(sub_inputs))
    
    # 7. 重组结果并生成分类标签
    labels = []
    ptr = 0
    
    # 结果包装进度条
    with tqdm(total=len(source_data), desc="📦 Packaging results", unit="data") as pkg_pbar:
        for data in source_data:
            for profile_idx, _ in enumerate(data['profile']):
                current_scores = log_probs[ptr:ptr+k]
                label_idx = current_scores.index(max(current_scores))
                labels.append(label_idx)
                ptr += k
            pkg_pbar.update(1)
            pkg_pbar.set_postfix({"profiles_packed": len(data['profile'])})
    
    return inputs, labels

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--cache_dir", default = "./cache")


if __name__ == "__main__":
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    opts = parser.parse_args()
    task = opts.task
    num_iterations = 3  # 获取迭代次数

    # 加载数据
    source_data_addr = f"data/{task}/train_questions.json"
    target_data_addr = f"data/{task}/train_outputs.json"
    with open(source_data_addr) as file:
        source_data = json.load(file)
    with open(target_data_addr) as file:
        target_data = json.load(file)["golds"]
    source_data = source_data[0:100]
    target_data = target_data[0:100]
    # 加载配置
    config_path = "configs/base.yaml"
    config = yaml.safe_load(open(config_path, 'r'))
    task_config = config["task_configurations"].get(task)
    k = 2  # 聚类数量

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classification_model_name = "bert-base-cased"
    classification_model = BertForClassification(classification_model_name, num_labels=k).to(device)
    classification_tokenizer = BertTokenizer.from_pretrained(classification_model_name, cache_dir=opts.cache_dir)

    generation_model_name = "Qwen/Qwen1.5-0.5B"
    generation_model = AutoModelForCausalLM.from_pretrained(generation_model_name, cache_dir=opts.cache_dir).to(device)
    generation_tokenizer = AutoTokenizer.from_pretrained(generation_model_name, cache_dir=opts.cache_dir)
    #generation_tokenizer.pad_token = generation_tokenizer.eos_token
    # 迭代训练循环
    print(f"Tokenizer实际max_length: {generation_tokenizer.model_max_length}")
    for iteration in range(num_iterations):
        print(f"\n=== 训练迭代 {iteration + 1}/{num_iterations} ===")
        
        # 1. 使用当前模型聚类
        print("🔄 执行深度聚类...")
        cluster_data = deep_cluster(
            source_data, 
            task, 
            task_config, 
            classification_model, 
            classification_tokenizer, 
            k
        )
        
        # 2. 生成训练数据
        print("⚙️ 生成增强数据...")
        inputs, labels = generate_data(
            generation_model,
            generation_tokenizer,
            source_data,
            cluster_data,
            target_data,
            task,
            task_config,
            k
        )
        
        from collections import Counter

        # 统计标签分布
        label_counts = Counter(labels)
        print("\n📊 标签分布统计:")
        for label, count in sorted(label_counts.items()):
            print(f"  类别 {label}: {count} 样本 ({count/len(labels):.1%})")

        # 3. 准备数据集
        print("📦 准备训练数据集...")
        def tokenize_function(examples):
            return classification_tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors=None
            )
        
        hf_dataset = Dataset.from_dict({"text": inputs, "labels": labels})
        tokenized_dataset = hf_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        tokenized_dataset.set_format(
            type="torch", 
            columns=["input_ids", "attention_mask", "labels"],
            output_all_columns=False
        )
        
        # 4. 训练分类模型
        print("🎯 训练分类模型...")
        training_args = TrainingArguments(
            output_dir=f"output/iter_{iteration}",      # 每次迭代保存到不同目录
            num_train_epochs=1,
            per_device_train_batch_size=4,
            logging_dir=f"logs/iter_{iteration}",
            logging_steps=50,
            save_strategy="no",
            learning_rate=5e-5,
            evaluation_strategy="no",
            report_to="none",
            weight_decay=0.01,
        )
        
        trainer = Trainer(
            model=classification_model,
            args=training_args,
            train_dataset=tokenized_dataset
        )
        trainer.train()
        model_save_path = os.path.join(training_args.output_dir, "full_model.pth")
        torch.save(classification_model, model_save_path)