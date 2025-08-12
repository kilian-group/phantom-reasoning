# Creating GSM infinite dataset

We create the dataset on G2 in the following way.

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
bash test_generate3.sh

mkdir -p data/gsm-infinite/zero_context/realistic
cp -r gsm_realistic/Igsm/zero_context/medium/ data/gsm-infinite/zero_context/realistic/
```