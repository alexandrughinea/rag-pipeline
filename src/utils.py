def clamp(value, min_limit=10, max_limit=50):
    return max(min(value, max_limit), min_limit)
