def check_missing_features(data, required_features):
    missing = [f for f in required_features if f not in data]
    return missing
