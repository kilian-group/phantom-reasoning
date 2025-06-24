# Multi-Hop QA with HotpotQA, 2WikiMultiHopQA, and MuSiQue

To evaluate on a dataset, please run:

```bash
python pred.py -od OUTPUT_DIR --dataset DATASET --split SPLIT --method METHOD --server vllm -m MODEL_NAME
```

`DATASET` can be one of

- `hp`
- `2wiki`
- `msq`
- `hp500`
- `2wiki500`
- `msq500`

The datasets with 500 in the name include a smaller dev split to make evaluation easier. When using `hp500`, `2wiki500`, and `msq500`, please set `SPLIT` to `minidev`.
The default `--data_dir` is `/share/nikola/phantom-reasoning/data`, which contains the JSON files for all datasets.

> \[!NOTE\]
> `pred.py` will automatically launch an LLM, so there is no need to run `vllm serve`.

To obtain the accuracy numbers, please run:

```bash
python format_split_accuracy.py -od OUTPUT_DIR --method METHOD --split SPLIT --dataset DATASET
```

## Additional Dependences

<details>
<summary>HotpotQA</summary>
```bash
pip install ujson
```
</details>
