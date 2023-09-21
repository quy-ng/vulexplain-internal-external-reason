import os
import click
import numpy as np
import evaluate
import torch
from project_dataset import load_dataset
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import Trainer, Seq2SeqTrainingArguments, EarlyStoppingCallback

rouge_metric = evaluate.load("rouge")

task_list = ['attack_vector', 'root_cause', 'impact', 'vulnerability_type']
max_src_length_configs = {
    'attack_vector':  1200,
    'root_cause': 1200,
    'impact': 1200,
    'vulnerability_type': 1200
}
max_des_length_configs = {
    'attack_vector': 146,
    'root_cause': 153,
    'impact': 167,
    'vulnerability_type': 53
}

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels

@click.command()
@click.option('-t', '--task_name', prompt='task name', type=click.Choice(task_list))
@click.option('--prexfix', default='t5p_script') 
@click.option('-w', '--size_weight', prompt='size of weight', type=click.Choice(['220', '770']))  # Choice must be string
@click.option('-s', '--save_dir', default='results')
@click.option('--train_fp16/--no-train_fp16', prompt='train with fp16', default=True)
@click.option('--nepoch', default=11)
@click.option('-b', '--batch', prompt='batch size', default=5)
@click.option('--ncpus', default=4)
def main(prexfix, task_name, size_weight, save_dir, train_fp16, nepoch, batch, ncpus):

    os.environ["WANDB_PROJECT"] = f"codet5p-{size_weight}m-{task_name}-{prexfix}"
    os.environ["WANDB_LOG_MODEL"] = "all"
    
    @dataclass
    class Args:
        model_name = f"Salesforce/codet5p-{size_weight}m"
        num_proc = ncpus
        batch_size = batch
        max_src_length = max_src_length_configs[task_name]
        max_des_length = max_des_length_configs[task_name]
        data_cols = ["CVE ID", "explain", "func_before", "processed_func"]
        output_dir = f'{save_dir}/{task_name}/{prexfix}_{size_weight}m'
        epochs = nepoch
        grad_acc_steps = 5
        lr = 5e-5
        fp16 = train_fp16
        lr_warmup_steps = 200
        weight_decay = 0.05
        task = task_name
    
    args = Args()

    codet5p_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)


    def preprocess_function(samples):
        source = samples["func_before"]
        target = samples["explain"]
        input_feature = codet5p_tokenizer(source, max_length=args.max_src_length, padding="max_length", truncation=True)
        labels = codet5p_tokenizer(target, max_length=args.max_des_length, padding="max_length", truncation=True)
        lables = labels["input_ids"].copy()
        lables = np.where(lables != codet5p_tokenizer.pad_token_id, lables, -100)
        return {  
                "input_ids": input_feature["input_ids"],
                "attention_mask": input_feature["attention_mask"],
                "labels": lables
        }
    
    def metrics_func(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions[0]
        pred_str = codet5p_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = codet5p_tokenizer.pad_token_id
        label_str = codet5p_tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        rouge_output = rouge_metric.compute(
            predictions=pred_str,
            references=label_str,
            rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
        )
        return {
            "R1": round(rouge_output["rouge1"], 4),
            "R2": round(rouge_output["rouge2"], 4),
            "RL": round(rouge_output["rougeL"], 4),
            "RLsum": round(rouge_output["rougeLsum"], 4),
        }

    dataset = load_dataset(args.task)
    tokenized_ds = dataset.map(
        preprocess_function,
        remove_columns=args.data_cols,
        batched=True,
        num_proc=args.num_proc,
        batch_size=args.batch_size)

    data_collator = DataCollatorForSeq2Seq(
      codet5p_tokenizer,
      model=model,
      return_tensors="pt")


    training_args = Seq2SeqTrainingArguments(
        report_to='wandb',
        output_dir=args.output_dir,
        do_train=True,
        do_predict=True,
        save_strategy='steps',
        save_steps=50,
        do_eval=True,
        metric_for_best_model='RL',
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        greater_is_better=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.lr_warmup_steps,
        save_total_limit=3,
        dataloader_num_workers=args.num_proc,
        fp16=args.fp16,
        logging_strategy="steps",
        logging_steps=50,
        auto_find_batch_size=True
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        data_collator = data_collator,
        compute_metrics = metrics_func,
        train_dataset = tokenized_ds["train"],
        eval_dataset = tokenized_ds["validation"],
        tokenizer = codet5p_tokenizer,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    predict_results = trainer.predict(tokenized_ds["test"])
    predictions = predict_results.predictions[0]
    metrics = predict_results.metrics
    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)
    predictions = np.where(predictions != -100, predictions, codet5p_tokenizer.pad_token_id)
    predictions = codet5p_tokenizer.batch_decode(predictions, skip_special_tokens=True)
    predictions = [pred.strip() for pred in predictions]
    output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
    with open(output_prediction_file, "w") as writer:
        writer.write("\n".join(predictions))

    

if __name__ == '__main__':
    main()
