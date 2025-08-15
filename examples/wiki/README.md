# Multi-Hop QA with HotpotQA, 2WikiMultiHopQA, and MuSiQue

## Setup Instructions

Please follow the main [README.md](../../README.md#setup-instructions).

## Results

1. Generate the predictions using the following command:

```bash
python pred.py -od OUTPUT_DIR --dataset DATASET --split SPLIT --method METHOD --server vllm -m MODEL_NAME
```

Here `DATASET` can be one of

- `hp`
- `2wiki`
- `msq`
- `hp500`
- `2wiki500`
- `msq500`

The datasets with 500 in the name include a smaller dev split to make evaluation easier. When using `hp500`, `2wiki500`, and `msq500`, please set `SPLIT` to `minidev`.

> \[!TIP\]
> On G2, you don't need to specify `--data_dir`. The default `--data_dir` is `/share/nikola/phantom-reasoning/data`, which contains the JSON files for all datasets.

> \[!NOTE\]
> `pred.py` will automatically launch an LLM, so there is no need to run `vllm serve`.

2. Compute accuracy using the following command:

```bash
python format_split_accuracy.py -od OUTPUT_DIR --method METHOD --split SPLIT --dataset DATASET
```

## Additional Setup Instructions for hp/hp500

```bash
pip install ujson
```
