# Creating GSM infinite dataset

We create the dataset on Unicorn in the following way.

1. Get CPU allocation

```bash
salloc -p kilian -n 8 -N 1 --mem=100GB -t 2-00:00:00
```

2. Create a conda environment and install dependencies

```bash
conda create -n gsm-infinite
conda activate gsm-infinite
conda install python=3.12

pip install uv
uv pip install -r requirements.txt
```

3. Run data generation script. This will output the json files in `./Igsm/zero_context/medium/{op}/*.jsonl`.
   And then we want to move the text into a specific location in `data/`.

```bash
bash generate_data.sh

mkdir -p data/gsm-infinite/zero_context/realistic
cp -r gsm_realistic/Igsm/zero_context/medium/ data/gsm-infinite/zero_context/realistic/
```

4. Run the data splitting script.
   This will split the generated data in `data/gsm-infinite` into training and evaluating dataset.

```bash
python gsm_realistic/split_datasets.py --data_dir data/ --seed 1
```

After splitting, these datasets are zipped at:

```bash
ls /share/nikola/phantom-reasoning/data/gsm-infinite-train.zip
ls /share/nikola/phantom-reasoning/data/gsm-infinite-eval.zip
```
