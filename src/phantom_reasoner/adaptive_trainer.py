from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union

from transformers import Trainer
import logging
import torch 

logger = logging.getLogger(__name__)

class AdaptiveTrainer(Trainer):
    """ Differences between AdaptiveTrainer and the regular Trainer classes
    Constructor additional arguments:
      1. Threshold which must be exceeded before going to next stage.
        - By default None -> trainer will move on when the evaluation accuracy plateaus 
        - This should be an edit in the _prepare_input function. However, I am also noticing


        HOLD ON: Maybe change this: get_batch_samples
        
        functions like _get_train_sampler, _get_eval_sampler
      2. Coefficient for the moving average (alpha)
        - For now, I am thinking of adding to the moving average by accessing the loss 
        in the compute_loss function, but I am not sure if this is the best way of doing this.
    
    train method:
      1. Stay at a level of difficulty for a until condition is reached
      2. When said condition is reached -> move onto the next difficulty level
    """

    def __init__(
        self,
        adaptive_threshold=None,
        moving_average_alpha=0.01,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.adaptive_threshold = adaptive_threshold
        self.moving_average_alpha = moving_average_alpha
        self.prev_metric = None
        self.train_moving_avg = 0
        self.val_moving_avg = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Calls super().compute_loss to compute the loss and then calculates the moving average using 
        the training loss. It has the same function as compute_loss in the super class, but also
        takes care of the moving average logic.
        """
        output = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        # Obtain loss depending on return_outputs and calculate moving average
        loss = output[0] if return_outputs else output 
        self.train_moving_avg = self.train_moving_avg * (1-self.moving_average_alpha) + loss * self.moving_average_alpha

        return output

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Calls super().log and follows by calculating the moving average using the validation loss.
        """
        super().log(logs, start_time)

        # Have to try this 
        print("", self.state.log_history[-1])

        self.state.log_history[-1] # has validation stuff

    def _should_move_to_next_difficulty(self):
        """
        Determines whether to move to the next difficulty level based on training metrics.
        
        Returns:
            bool: True if we should move to the next difficulty level, False otherwise.
        """
        # Get current loss
        current_loss = self._total_loss_scalar / max(self.state.global_step, 1)
        
        # Update moving average
        if self.moving_avg is None:
            self.moving_avg = current_loss
        else:
            self.moving_avg = self.ma_alpha * self.moving_avg + (1 - self.ma_alpha) * current_loss
        
        logger.info(f"Current loss: {current_loss}, Moving average: {self.moving_avg}")
        
        # If threshold is provided, check if we've reached it
        if self.adaptive_threshold is not None:
            return self.moving_avg <= self.adaptive_threshold
        else:
            # If no threshold is provided, check if the loss has plateaued
            # by comparing with the previous metric
            if self.prev_metric is None:
                self.prev_metric = self.moving_avg
                return False
            
            # Calculate relative improvement
            improvement = (self.prev_metric - self.moving_avg) / self.prev_metric
            
            # Update previous metric
            self.prev_metric = self.moving_avg
            
            # If improvement is less than 1%, consider it plateaued
            return improvement < 0.01  # 1% improvement threshold
        

class AdaptivePW(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.max_difficulty = max([Q_A['difficulty'] for Q_A in base_dataset])
        self.current_difficulties = [0]
        self.current_indices = self._get_indices_for_difficulty(self.current_difficulties)
        
    def _get_indices_for_difficulty(self, difficulties):
        """Get indices for the current difficulty level."""
        indices = [i for i, Q_A in enumerate(self.base_dataset) if Q_A['difficulty'] in difficulties]
        return indices
    
    def set_difficulty(self, difficulties):
        """Set the current difficulty level."""
        self.current_difficulty = [min(difficulty, self.max_difficulty) for difficulty in difficulties]
        self.current_indices = self._get_indices_for_difficulty(self.current_difficulty)
    
    def __len__(self):
        return len(self.current_indices)
    
    def __getitem__(self, idx):
        if idx >= len(self.current_indices):
            raise IndexError(f"Index {idx} out of range for difficulty {self.current_difficulty}")
        base_idx = self.current_indices[idx]
        return self.base_dataset[base_idx]