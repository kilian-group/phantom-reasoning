import matplotlib.pyplot as plt

# utils for plotting
plt.rcParams.update({"font.family": "Fira Code"})

COLORS2HEX = {
    "myYellow": "#E8A93C",
    "myOrange": "#D96831",
    "myGreen": "#3B9B7B",
    "myBlue": "#3B8FBF",
    "myPurple": "#8B5FA8",
}

TRAIN_DATASET_NAMES = ["base", "format", "gsminf", "pw", "rg-family_relationships", "rg-knights_knaves"]
TRAIN_DATASET_ALIAS2NAME = {
    "base": "base",
    "format": "format",
    "gsminf": "GSM-$\\infty$",
    "pw": "PhantomWiki",
    "rg-family_relationships": "RG-Family",
    "rg-knights_knaves": "RG-Knights",
}
TRAIN_DATASET_ALIAS2COLOR = {
    "base": "myYellow",
    "gsminf": "myBlue",
    "pw": "myOrange",
    "rg-family_relationships": "myYellow",
    "rg-knights_knaves": "myPurple",
}

EVAL_DATASET_NAMES = ["HotpotQA", "2Wiki", "MuSiQue", "CofCA", "SynthWorlds-RM"]

# Single column figures
MARKER_ALPHA = 1
MARKER_SIZE = 3
LINE_ALPHA = 0.75
OUTWARD = 4
LINE_WIDTH = 1

LABEL_FONT_SIZE = 13
MINOR_TICK_FONT_SIZE = 5
TICK_FONT_SIZE = 10
LEGEND_FONT_SIZE = 13

MODEL_NAME2ALIAS = {
    "microsoft/Phi-4-mini-reasoning": "Phi-4-Mini-Reasoning",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-0.6B": "Qwen3-0.6B",
    "Qwen/Qwen3-1.7B": "Qwen3-1.7B",
    "Qwen/Qwen3-4B": "Qwen3-4B",
}

TRAIN_DATASET_NAME2MARKER = {
    "gsminf": "s",
    "pw": "o",
    "rg-family_relationships": "d",
    "rg-knights_knaves": "v",
}
