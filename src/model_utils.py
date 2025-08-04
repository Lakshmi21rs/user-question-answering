from transformers import BertForQuestionAnswering, BertTokenizerFast

def load_base_model():
    model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    return model, tokenizer

def load_finetuned_model(model_dir="./models/qa_model"):
    model = BertForQuestionAnswering.from_pretrained(model_dir)
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    return model, tokenizer
