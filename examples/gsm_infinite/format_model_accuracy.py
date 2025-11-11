import argparse
from pathlib import Path

import pandas as pd
from tabulate import tabulate

# Directory containing result files
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", "-od", type=str, default="out")
args = parser.parse_args()

# Load all CSV files in the output directory
# Print accuracy grouped by model

all_df = []
scores_path = Path(args.output_dir) / Path("scores")
for csv_file_path in scores_path.glob("*.csv"):
    df = pd.read_csv(csv_file_path)
    all_df.append(df)
all_df = pd.concat(all_df)
acc = all_df.groupby("model_name")["num_correct"].agg("sum") / all_df.groupby("model_name")["count"].agg(
    "sum"
)
acc = acc.reset_index()
print(tabulate(acc, headers="keys", tablefmt="github"))
