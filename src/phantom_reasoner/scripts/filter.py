"""
Notes:
  - Make sure to cd into `src`
  - Make sure to not add a slash at the end of `traces-path`
  - Ensure that `traces-path` folder starts with "out-v" 
Example usage:
  - python -m phantom_reasoner.scripts.filter --metric f1 --threshold 0.9 --traces-path "/share/nikola/phantom-wiki/eval/out-v05-0222"
  - This will filter traces based on the given metric and threshold, e.g. f1>=0.9, and will save the results to 
  the output path, in this case "/share/nikola/phantom-wiki/eval/out-v05-0222-filtered-f1-above-0.9", and mimick the json file structure.
"""

# Importing all libraries
import os
import json
import argparse
from tqdm import tqdm
from phantom_reasoner.utils.score import f1, recall, precision, exact_match


def filter_traces(prediction_dir, metric, threshold=1):
  """
  Extracts a subset of model prediction traces that meet a specified quality threshold.

  This function walks through a directory structure containing model predictions,
  evaluates each prediction against its ground truth using the specified metric,
  and collects those that meet or exceed the given threshold.

  Parameters
  ----------
  prediction_dir : str
      Path to the directory containing subdirectories for different models.
      Each model subdirectory should contain JSON files with prediction traces.
  
  metric : str
      The evaluation metric to use. Must be one of:
      - "f1": F1 score (harmonic mean of precision and recall)
      - "recall": Recall score (ratio of correctly predicted positive observations to all actual positives)
      - "precision": Precision score (ratio of correctly predicted positive observations to total predicted positives)
      - "em": Exact match (binary score: 1 if prediction exactly matches ground truth, 0 otherwise)
  
  threshold : float, default=1
      The minimum score required to consider a trace valid.
      For "em" metric, any non-zero value means the prediction is an exact match.
      For other metrics, predictions must achieve at least this score to be included.

  Returns
  -------
  dict
      A nested dictionary with the following structure:
      {
          "model_directory_path": {
              "prediction_file_name": {
                  "query_id": {
                      "true": [...],  # List of ground truth labels
                      "pred": [...],  # List of model predictions
                      ...             # Other trace information from the original file
                  },
                  ...
              },
              ...
          },
          ...
      }
      Only traces meeting the threshold criteria are included.

  Notes
  -----
  - Hidden files (starting with '.') in the prediction directory are ignored.
  - Each prediction trace file should be a JSON file containing a dictionary
    where keys are query IDs and values are dictionaries with at least 'true'
    and 'pred' keys containing ground truth and prediction data.
  - For the "em" metric, the score itself is used as the validity check (1 = valid).
    For other metrics, the score must be >= the threshold.
  """
  prediction_dir = os.path.join(prediction_dir, "preds") # All out repos contain "preds" by default in our phantom-eval implementation

  # Get all method directories, filtering out hidden files/directories
  method_dirs = [os.path.join(prediction_dir, i) for i in os.listdir(prediction_dir)
                if not i.startswith('.')]
  
  # Initialize dictionary to store valid traces for all methods
  all_valid_traces = {}

  for md in tqdm(method_dirs, desc="Filtering traces"): # Process each model directory with a progress bar
    # Get all prediction trace files for this model
    pred_trace_files = os.listdir(md)
    model_valid_traces = {}

    # Process each prediction trace file
    for pt_file in pred_trace_files:
      pt_file_path = os.path.join(md, pt_file)
      valid_traces = {}

      # Load prediction traces from JSON file
      with open(pt_file_path) as f:
        preds_traces = json.load(f)
      
      # Iterate through each trace (question)
      for qi, single_pred_trace in preds_traces.items():
        # Extract ground truth and predictions
        labels = ", ".join(single_pred_trace['true'])
        preds = single_pred_trace['pred']

        # Calculate the appropriate evaluation metric
        if metric=="f1":
          score = f1(preds, labels)
        elif metric=="recall":
          score = recall(preds, labels)
        elif metric=="precision":
          score = precision(preds, labels)
        elif metric=="em":
          score = exact_match(preds, labels)

        # For exact match, we either have True or False, thus (EM>=1) == EM
        # For other metrics, score must meet or exceed threshold
        is_trace_valid = score>=threshold

        # If the trace meets criteria, add it to the valid traces
        if is_trace_valid:
          valid_traces[qi] = single_pred_trace

      # Store valid traces for this prediction file
      model_valid_traces[pt_file] = valid_traces

    # Store valid traces for this model
    all_valid_traces[md] = model_valid_traces
  
  return all_valid_traces

def save_filtered_traces(filtered_traces, output_dir, metric, threshold):
    """
    Saves the filtered prediction traces to a specified directory, maintaining the original file structure.
    
    This function takes the output of obtain_subset_of_traces() and saves it to disk,
    preserving the same directory structure as the original prediction traces.
    The output directory will be named according to the filter criteria (metric and threshold).
    
    Parameters
    ----------
    filtered_traces : dict
        The nested dictionary returned by obtain_subset_of_traces(), containing filtered
        prediction traces organized by model directory and prediction file.
    
    output_dir : str
        Base directory where the filtered traces will be saved.
        The function will create a subdirectory with an appropriate name based on metric and threshold.
    
    metric : str
        The evaluation metric used for filtering (f1, recall, precision, or em).
        Used to generate an appropriate output directory name.
    
    threshold : float
        The threshold value used for filtering.
        Used to generate an appropriate output directory name.
    
    Returns
    -------
    str
        Path to the created output directory.
    
    Notes
    -----
    - For exact match (em) with threshold=1, the directory will be named "only-correct"
    - For other metrics, the directory will indicate the metric and threshold
      (e.g., "filtered-f1-above-0.9" for F1 >= 0.9)
    - The function preserves the relative paths of model directories and files
    """
    # Create appropriate output directory name based on filtering criteria
    if metric == "em":
        filter_dir_name = "-only-correct"
    elif threshold==1:
        filter_dir_name = f"-filtered-{metric}-correct"
    else:
        filter_dir_name = f"-filtered-{metric}-above-{threshold}"

    unfiltered_dir_name = os.path.basename(output_dir) # Used to obtain version number
    filter_dir_name = "out-v" + unfiltered_dir_name.split("out-v")[1] + filter_dir_name

    # Construct the full output directory path
    base_dir = os.path.dirname(output_dir) 
    full_output_dir = os.path.join(os.path.join(base_dir, filter_dir_name), "preds") # Save to preds to stay consistent with norms
    
    # Process each method directory
    for method_dir_path, method_traces in tqdm(filtered_traces.items(), desc="Saving filtered traces"):
        # Extract the model directory name from the full path
        method_dir_name = os.path.basename(method_dir_path)
        
        # Create the corresponding output model directory
        method_dir_name = os.path.join(full_output_dir, method_dir_name)
        os.makedirs(method_dir_name, exist_ok=True)
        
        # Save each prediction file for this model
        for pred_file_name, valid_traces in method_traces.items():
            # Skip if there are no valid traces for this file
            if not valid_traces:
                continue
            
            # Create the output file path
            output_file_path = os.path.join(method_dir_name, pred_file_name)
            
            # Save the filtered traces to a JSON file
            with open(output_file_path, 'w') as f:
                json.dump(valid_traces, f, indent=4)
                
    print(f"Filtered traces saved to: {full_output_dir}")
    return full_output_dir



if __name__=="__main__":
  parser = argparse.ArgumentParser(
      description="Filter prediction traces based on quality metrics."
  )
  parser.add_argument("--metric", type=str, required=True, choices=["f1", "precision", "recall", "em"])  
  parser.add_argument("--threshold", type=float, required=True)
  parser.add_argument("--traces-path", type=str, required=True)

  args = parser.parse_args()
  filtered_traces = filter_traces(args.traces_path, args.metric, args.threshold)
  save_filtered_traces(filtered_traces, args.traces_path, args.metric, args.threshold)
