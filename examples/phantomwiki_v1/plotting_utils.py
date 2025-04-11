from phantom_eval.plotting_utils import *  # noqa: F401, F403

# Override the default method line styles
METHOD_LINESTYLES = {
    "cot": "solid",
    "grpo": "dashed",
    # TODO: add line styles for our curriculum learning methods here
}
# Override the default method aliases
METHOD_ALIASES = {
    "cot": "CoT",
    "grpo": "GRPO",
    # TODO: add aliases for our curriculum learning methods here
}

# NOTE: comment out the methods you don't want to plot
INCONTEXT_METHODS = [
    "cot",  # CoTAgent
    "grpo",
    # TODO: add our curriculum learning methods here
]

DEFAULT_MODEL_LIST = [
    "qwen/qwen2.5-1.5b-instruct",
    "meta-llama/llama-3.2-1b-instruct",
    "deepseek-ai/deepseek-r1-distill-qwen-1.5b",
]
