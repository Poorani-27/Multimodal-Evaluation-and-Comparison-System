# cache.py
def cache_query(cache, text, task):
    return cache.get((text, task))
