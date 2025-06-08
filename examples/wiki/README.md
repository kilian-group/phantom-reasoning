# Multi-Hop QA with HotpotQA, 2WikiMultiHopQA, and MuSiQue

To evaluate on HotpotQA, please run:

```bash
python pred.py -od OUTPUT_DIR --dataset hp --split SPLIT --method METHOD --server vllm -m MODEL_NAME -bs 100
```

To evaluate on 2WikiMultiHopQA, set `--dataset` to `2wiki`; to evaluate on MuSiQue, set `--dataset` to `msq`. The default `--data_dir` is `/share/nikola/phantom-reasoning/data`, which contains the json files for these three datasets.

> \[!TIP\]
> Since we are using vLLM, use a large batch size (e.g., 100) to maximimize throughput.

## Additional Dependences

<details>
<summary>HotpotQA</summary>
```bash
pip install ujson
```
</details>
