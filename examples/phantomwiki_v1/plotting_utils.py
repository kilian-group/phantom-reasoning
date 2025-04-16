from phantom_eval.plotting_utils import *  # noqa: F401, F403

METHOD_LATEX_ALIASES = {
    "zeroshot": "\\zeroshot",
    "cot": "\\CoT",
    "grpo": "GRPO",
    "zeroshot-rag": "\\zeroshotrag",
    "cot-rag": "\\cotrag",
    "act": "\\act",
    "react": "\\react",
}

# Override the default method line styles
METHOD_LINESTYLES = {
    "zeroshot": "dashed",
    "cot": "solid",
    "grpo": "dashed",
    # TODO: add line styles for our curriculum learning methods here
}
# Override the default method aliases
METHOD_ALIASES = {
    "zeroshot": "ZeroShot",
    "cot": "CoT",
    "grpo": "GRPO",
    # TODO: add aliases for our curriculum learning methods here
}

# NOTE: comment out the methods you don't want to plot
INCONTEXT_METHODS = [
    "zeroshot",
    "cot",
    # "grpo",
    # TODO: add our curriculum learning methods here
]

DEFAULT_MODEL_LIST = [
    "qwen/qwen2.5-1.5b-instruct",
    "meta-llama/llama-3.2-1b-instruct",
    "deepseek-ai/deepseek-r1-distill-qwen-1.5b",
]
