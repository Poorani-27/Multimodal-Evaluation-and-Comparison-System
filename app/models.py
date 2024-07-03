# models.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

def load_models():
    models = {
        "bert_base": {
            "model": AutoModelForSequenceClassification.from_pretrained("bert-base-uncased"),
            "tokenizer": AutoTokenizer.from_pretrained("bert-base-uncased")
        },
        "roberta": {
            "model": AutoModelForSequenceClassification.from_pretrained("roberta-base"),
            "tokenizer": AutoTokenizer.from_pretrained("roberta-base")
        },
        "distilbert": {
            "model": AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased"),
            "tokenizer": AutoTokenizer.from_pretrained("distilbert-base-uncased")
        },
        "albert": {
            "model": AutoModelForSequenceClassification.from_pretrained("albert-base-v2"),
            "tokenizer": AutoTokenizer.from_pretrained("albert-base-v2")
        },
        "xlnet": {
            "model": AutoModelForSequenceClassification.from_pretrained("xlnet-base-cased"),
            "tokenizer": AutoTokenizer.from_pretrained("xlnet-base-cased")
        }
    }
    
    # Load pipelines for specific tasks
    models["bert_base"]["pipeline"] = pipeline("sentiment-analysis", model=models["bert_base"]["model"], tokenizer=models["bert_base"]["tokenizer"])
    models["roberta"]["pipeline"] = pipeline("sentiment-analysis", model=models["roberta"]["model"], tokenizer=models["roberta"]["tokenizer"])
    models["distilbert"]["pipeline"] = pipeline("sentiment-analysis", model=models["distilbert"]["model"], tokenizer=models["distilbert"]["tokenizer"])
    models["albert"]["pipeline"] = pipeline("sentiment-analysis", model=models["albert"]["model"], tokenizer=models["albert"]["tokenizer"])
    models["xlnet"]["pipeline"] = pipeline("sentiment-analysis", model=models["xlnet"]["model"], tokenizer=models["xlnet"]["tokenizer"])
    
    return models
