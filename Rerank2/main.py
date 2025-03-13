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

parser = argparse.ArgumentParser()

parser.add_argument("--task", required = True)
parser.add_argument("--experiment_name", required = True)
parser.add_argument("--max_length", type = int, default = 256)
parser.add_argument("--generation_max_length", type = int, default = 128)
parser.add_argument("--per_device_batch_size", type = int, default = 16)
parser.add_argument("--generation_num_beams", type = int, default = 4)
parser.add_argument("--cache_dir", default = "./cache")



if __name__ == "__main__":

    opts = parser.parse_args()
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small", cache_dir=opts.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", cache_dir=opts.cache_dir)
    collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = model, max_length = opts.max_length)

    output_dir = "experiments/{}/{}".format(opts.experiment_name, opts.task)

    task = opts.task
    prompt_generator = simple_prompt_generator(opts.max_length, tokenizer)
    
    source_data_addr = "data/{}/dev_questions.json".format(task)
    target_data_addr = "data/{}/dev_outputs.json".format(task)



    with open(source_data_addr) as file:
        source_data = json.load(file)
    
    with open(target_data_addr) as file:
        target_data = json.load(file)["golds"]

    if task == "LaMP-1":
        labels = get_all_labels(task)
        eval_dataset = SimpleDataset(source_data=source_data, target_data=target_data, task=task, create_prompt=prompt_generator)
        compute_metrics = create_metric_f1_accuracy(tokenizer = tokenizer, all_labels = labels)
    elif task == "LaMP-2":
        labels = get_all_labels(task)
        eval_dataset = SimpleDataset(source_data=source_data, target_data=target_data, task=task, create_prompt=prompt_generator)
        compute_metrics = create_metric_f1_accuracy(tokenizer = tokenizer, all_labels = labels)
    elif task == "LaMP-3":
        labels = get_all_labels(task)
        eval_dataset = SimpleDataset(source_data=source_data, target_data=target_data, task=task, create_prompt=prompt_generator)
        compute_metrics = create_metric_mae_rmse(tokenizer = tokenizer, all_labels = labels)
    elif task == "LaMP-4":
        eval_dataset = SimpleDataset(source_data=source_data, target_data=target_data, task=task, create_prompt=prompt_generator)
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer = tokenizer)
    elif task == "LaMP-5":
        eval_dataset = SimpleDataset(source_data=source_data, target_data=target_data, task=task, create_prompt=prompt_generator)
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer = tokenizer)
    elif task == "LaMP-7":
        eval_dataset = SimpleDataset(source_data=source_data, target_data=target_data, task=task, create_prompt=prompt_generator)
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer = tokenizer)
    elif task == "LaMP-6":
        eval_dataset = SimpleDataset(source_data=source_data, target_data=target_data, task=task, create_prompt=prompt_generator)
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer = tokenizer)
   
    eval_dataset = convert_to_hf_dataset(eval_dataset, cache_dir = opts.cache_dir).map(create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length), batched=True)

    


    training_args = Seq2SeqTrainingArguments(
        output_dir = output_dir,
        do_eval = True,
        per_device_eval_batch_size = opts.per_device_batch_size,
        generation_num_beams = opts.generation_num_beams,
        predict_with_generate = True,
        eval_accumulation_steps = 1,
        generation_max_length = opts.generation_max_length
    )

    trainer = Seq2SeqTrainer(
        model = model,
        args = training_args,
        data_collator = collator,
        eval_dataset = eval_dataset,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics
    )


    results = trainer.evaluate(eval_dataset)
    print(results)

