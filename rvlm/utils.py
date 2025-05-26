import json
import pickle


def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def read_json_or_dict(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except:
        data = {}
    return data

def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent='\t')

def read_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def write_pkl(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


