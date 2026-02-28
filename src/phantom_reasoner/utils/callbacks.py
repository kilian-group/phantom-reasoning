import logging
import os
import shutil
import subprocess
from pathlib import Path

from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.training_args import TrainingArguments

from phantom_reasoner.configs import GRPOScriptArguments

logger = logging.getLogger(__name__)


CALLBACKS: dict[str, TrainerCallback] = {}


def get_callbacks(train_config, model_config) -> list[TrainerCallback]:
    callbacks = []
    for callback_name in train_config.callbacks:
        if callback_name not in CALLBACKS:
            raise ValueError(f"Callback {callback_name} not found in CALLBACKS.")
        callbacks.append(CALLBACKS[callback_name](model_config))

    return callbacks


##################################
# END Callbacks that take in ModelConfig used by SFT trainer
##################################


class DeleteAllButLastOptimizerCheckpointCallback(TrainerCallback):
    """Callback to delete all optimizer states except the last checkpoint."""

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Delete all optimizer checkpoints except the last one
        glob_checkpoints = [
            str(x) for x in Path(args.output_dir).glob(f"{PREFIX_CHECKPOINT_DIR}-*") if os.path.isdir(x)
        ]
        last_checkpoint = str(Path(args.output_dir).joinpath(f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"))
        for checkpoint in glob_checkpoints:
            if checkpoint != last_checkpoint:
                # Delete optimizer state in directory checkpoint/global_step*
                glob_optimizer_states = [str(x) for x in Path(checkpoint).glob("global_step*")]
                for optimizer_state in glob_optimizer_states:
                    logger.info(f"Deleting optimizer state {optimizer_state}")
                    shutil.rmtree(optimizer_state, ignore_errors=True)

        return control


class PhantomEvalCallback(TrainerCallback):
    """Callback to run phantom_eval on saving a checkpoint."""

    def __init__(self, script_args: GRPOScriptArguments):
        """
        Args:
            script_args (GRPOScriptArguments): Script arguments containing evaluation parameters.
        """
        self.script_args = script_args

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        eval_out_dir = os.path.join(args.output_dir, "out")

        # Run phantom_eval on the saved checkpoint ONLY on the first GPU
        pw_eval_cmd = [
            "python",
            "-m",
            "phantom_eval",
            "--method",
            self.script_args.prompt_method,
            "--server",
            "vllm",
            "--inf_vllm_offline",
            "--model_name",
            checkpoint_folder,
            "--dataset",
            self.script_args.eval_dataset_name,
            "--split_list",
            *self.script_args.eval_split_list,
            "--inf_vllm_tensor_parallel_size",
            "1",
            "--inf_vllm_max_model_len",
            str(
                args.max_prompt_length + args.max_completion_length
            ),  # eval should use the same max prompt length as training
            "-od",
            eval_out_dir,
        ]
        if self.script_args.eval_from_local:
            pw_eval_cmd.append("--from_local")

        if self.script_args.ignore_think_tags_in_outputs:
            # NOTE: in phantom_eval, this flag is used to ignore <think> tags in outputs
            pw_eval_cmd.append("--inf_is_deepseek_r1_model")

        if self.script_args.exclude_aggregation_questions:
            pw_eval_cmd.append("--exclude_aggregation_questions")

        if args.should_save:
            # In multi-GPU training, we only run phantom_eval from the main process
            # that was trying to save the checkpoint. This ensures that we call
            # phantom_eval only once per checkpoint, not once per GPU.
            env = PhantomEvalCallback.get_env_vars_for_pw_eval_vllm()

            try:
                # Block until the PhantomEval process finishes
                # TODO: think about non-blocking, but we need to be careful since
                # older checkpoints will be deleted during training (only a number of
                # checkpoints are retained)
                logger.info(f"*** Running PhantomEval with command: {' '.join(pw_eval_cmd)}")
                _ = subprocess.run(pw_eval_cmd, env=env, check=True, text=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"!!! PhantomEval failed with error:\n{e.stderr}")
                raise e

        return control

    @staticmethod
    def get_env_vars_for_pw_eval_vllm() -> dict[str, str]:
        """Get environment variables for running PhantomEval with vLLM."""
        # Set vllm visible devices to the first GPU
        env = {"CUDA_VISIBLE_DEVICES": "0"}
        # Copy all env vars that contain "CONDA" or "PYTHON" or "PATH"
        # These are needed by the subprocess to launch vllm and phantom_eval
        for key, value in os.environ.items():
            if "CONDA" in key or "PYTHON" in key or "PATH" in key:
                env[key] = value
        return env
