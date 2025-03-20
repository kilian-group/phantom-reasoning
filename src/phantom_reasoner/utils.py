import os
import json

# TODO: remove hardcoding later
BASE_DIR = "/share/nikola/phantom-wiki/eval/out-v05-0222-filtered-f1-above-0.9/preds/"

def filter_by_split_model(preds_dir:str, split: str, modelname: str):
    """
    filter the predictions by split and model
    Args:       
        dir: the directory where the predictions are stored as .json files
            each json file is a dictionary in the format of 
            {"question_id": {
                "interaction": {"messages": list[{"role": str, "content": str}]},
                ...
                }
            }
        split: the split of PhantomWiki to filter by
        model: the model that made the predictions
    Returns:
        a dictionary containing the filtered predictions
    """
    dir = os.path.join(BASE_DIR, preds_dir)
    filtered_predictions = {}
    for file in os.listdir(dir):
        if file.endswith(".json"):
            if file.startswith(f"split={split}") and f"model_name={modelname}" in file:
                with open(os.path.join(dir, file), "r") as f:
                    # load the predictions from the file and add them to the dictionary
                    data = json.load(f)
                    filtered_predictions.update(data)
    return filtered_predictions
