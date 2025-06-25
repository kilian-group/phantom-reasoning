import os
from datetime import datetime


def get_run_name(training_algo_name: str, script_args, model_args, run_flags_str: str = "") -> str:
    """
    Returns <run_dir>/<dataset_name>/<model_name>/<training_algo_name>/$USER/<MMDD>__<run_flags_str>

    NOTE: `script_args` must have attributes: `dataset_name`, `run_dir`.
    NOTE: `model_args` must have attribute `model_name_or_path`.
    If `run_flags_str` is not provided, the preceding "__" will be omitted.
    """
    assert hasattr(script_args, "run_dir"), "script_args must have attribute run_dir"
    assert hasattr(script_args, "dataset_name"), "script_args must have attribute dataset_name"
    assert hasattr(model_args, "model_name_or_path"), "model_args must have attribute model_name_or_path"

    user = str(os.getenv("USER"))
    today = datetime.now().strftime("%m%d")
    run_name = (
        f"{script_args.run_dir}/{script_args.dataset_name}/{model_args.model_name_or_path}/"
        + f"{training_algo_name}/{user}/{today}"
    )
    if run_flags_str:
        run_name += f"__{run_flags_str}"
    return run_name
