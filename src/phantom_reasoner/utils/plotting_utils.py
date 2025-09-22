import matplotlib.pyplot as plt

# utils for plotting
plt.rcParams.update({"font.family": "Fira Code"})

COLORS2HEX = {"myYellow": "#E8A93C", "myOrange": "#D96831", "myGreen": "#3B9B7B", "myBlue": "#3B8FBF"}
models = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen2.5-1.5B-Instruct"]

TRAIN_DATASET_NAMES = ["base", "format", "gsminf", "pw"]
TRAIN_DATASET_ALIAS2NAME = {
    "base": "base",
    "format": "format",
    "gsminf": "GSM-$\\infty$",
    "pw": "PhantomWiki",
}
TRAIN_DATASET_ALIAS2COLOR = {"base": "myYellow", "format": "myOrange", "gsminf": "myGreen", "pw": "myBlue"}

EVAL_DATASET_NAMES = ["HotpotQA", "2Wiki", "MuSiQue"]

# Single column figures
MARKER_ALPHA = 1
MARKER_SIZE = 3
LINE_ALPHA = 0.75
OUTWARD = 4

LABEL_FONT_SIZE = 13
MINOR_TICK_FONT_SIZE = 5
TICK_FONT_SIZE = 10
LEGEND_FONT_SIZE = 13

MODEL_ALIASES = {
    "microsoft/Phi-4-mini-reasoning": "Phi-4-Mini-Reasoning",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen3-0.6B": "Qwen3-0.6B",
    "Qwen/Qwen3-1.7B": "Qwen3-1.7B",
}
