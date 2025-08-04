from transformers import BertForQuestionAnswering, BertTokenizerFast
import torch

def predict_answer(question, context, model, tokenizer):
    inputs = tokenizer(question, context, return_tensors='pt', max_length=512, truncation=True)
    outputs = model(**inputs)

    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits) + 1

    answer_ids = inputs["input_ids"][0][start:end]
    return tokenizer.decode(answer_ids, skip_special_tokens=True)
