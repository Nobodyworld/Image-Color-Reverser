# /utils/config_manager.py
import json

def load_config(config_path='config/config.json'):
    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
            return config
    except FileNotFoundError:
        raise Exception("config.json file not found.")
    except json.JSONDecodeError:
        raise Exception("Failed to decode config.json.")