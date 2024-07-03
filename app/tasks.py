# tasks.py
def process_text(models, text, task):
    results = {}
    
    for model_name, model_info in models.items():
        pipeline = model_info["pipeline"]
        result = pipeline(text)
        results[model_name] = result
    
    return results
