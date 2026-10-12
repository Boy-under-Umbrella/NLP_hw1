from split_utils import SEED, split_nyt, tokenize_nltk, doc_avg_embedding, load_glove
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch

model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# label映射，把字符串label转数字
labels_list = sorted(list(df_train["label"].unique()))
label2id = {lab:i for i,lab in enumerate(labels_list)}
id2label = {i:lab for i,lab in enumerate(labels_list)}
num_labels = len(labels_list)

def tokenize_func(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)

# 转huggingface Dataset格式
from datasets import Dataset
ds_train = Dataset.from_pandas(df_train)
ds_val = Dataset.from_pandas(df_val)
ds_test = Dataset.from_pandas(df_test)

# label映射成数字
ds_train = ds_train.map(lambda x: {"labels": label2id[x["label"]]})
ds_val = ds_val.map(lambda x: {"labels": label2id[x["label"]]})
ds_test = ds_test.map(lambda x: {"labels": label2id[x["label"]]})

ds_train_tok = ds_train.map(tokenize_func, batched=True)
ds_val_tok = ds_val.map(tokenize_func, batched=True)
ds_test_tok = ds_test.map(tokenize_func, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels, label2id=label2id, id2label=id2label
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macrof1 = f1_score(labels, preds, average="macro")
    return {"accuracy":acc, "macro_f1":macrof1}

training_args = TrainingArguments(
    output_dir="./bert_output",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    seed=SEED,
    load_best_model_at_end=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train_tok,
    eval_dataset=ds_val_tok,
    compute_metrics=compute_metrics
)

# 训练
trainer.train()

# 在NYT测试集上评估
test_result = trainer.evaluate(ds_test_tok)
print("BERT Test Accuracy:", test_result["eval_accuracy"])
print("BERT Test Macro‑F1:", test_result["eval_macro_f1"])