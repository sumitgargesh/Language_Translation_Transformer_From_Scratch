from pathlib import Path



'''
    Saving the weights of the model
'''
def get_config():
    return{
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weigts_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basemane = config['model_basename']
    model_filename = f"{model_basemane}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)