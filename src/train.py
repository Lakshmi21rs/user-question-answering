from transformers import TrainingArguments, Trainer
from datasets import Dataset
import torch

def prepare_dataset(df, tokenizer):
    def add_token_positions(example):
        inputs = tokenizer(
            example['question'],
            example['context'],
            truncation=True,
            padding="max_length",
            max_length=512
        )
        start = example['context'].find(example['answer'])
        end = start + len(example['answer'])

        inputs['start_positions'] = start
        inputs['end_positions'] = end
        return inputs

    dataset = Dataset.from_pandas(df)
    return dataset.map(add_token_positions)

def train_model(model, tokenizer, dataset):
    training_args = TrainingArguments(
        output_dir="./models/qa_model",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_steps=10,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()
    model.save_pretrained('./models/qa_model')
    tokenizer.save_pretrained('./models/qa_model')
