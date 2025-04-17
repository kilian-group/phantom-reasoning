from transformers import Trainer
import logging
import torch 

logger = logging.getLogger(__name__)

class AdaptiveTrainer(Trainer):
    """ Differences between AdaptiveTrainer and the regular Trainer classes
    Constructor additional arguments:
      1. Threshold which must be exceeded before going to next stage.
        - By default None -> trainer will move on when the evaluation accuracy plateaus 
      2. Coefficient for the moving average (alpha)
    
    train method:
      1. Stay at a level of difficulty for a until condition is reached
      2. When said condition is reached -> move onto the next difficulty level
    """

    def __init__(
        self,
        adaptive_threshold=None,
        ma_alpha=0.9,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.adaptive_threshold = adaptive_threshold
        self.ma_alpha = ma_alpha
        self.prev_metric = None
        self.moving_avg = None

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        """
        Main training entry point.
        
        This method extends the default Trainer.train method to implement adaptive training,
        where we stay at a certain level of difficulty until a condition is met.
        """
        # Start with difficulty level 0
        current_difficulty = 0
        
        # Set up data source with initial difficulty
        if hasattr(self.train_dataset, 'set_difficulty'):
            self.train_dataset.set_difficulty(current_difficulty)
        else:
            logger.warning("Training dataset does not implement set_difficulty method. Adaptive training may not work correctly.")
        
        # Initialize moving average for tracking progress
        self.moving_avg = None
        
        max_difficulty = getattr(self.train_dataset, 'max_difficulty', 0)
        
        while current_difficulty <= max_difficulty:
            logger.info(f"Training at difficulty level {current_difficulty}")
            
            # Call the original training loop with the current difficulty
            result = super().train(resume_from_checkpoint, trial, ignore_keys_for_eval)
            
            # Check if we should move to the next difficulty
            if current_difficulty < max_difficulty and self._should_move_to_next_difficulty():
                current_difficulty += 1
                logger.info(f"Moving to difficulty level {current_difficulty}")
                if hasattr(self.train_dataset, 'set_difficulty'):
                    self.train_dataset.set_difficulty(current_difficulty)
                # Reset the moving average for the new difficulty level
                self.moving_avg = None
            else:
                logger.info(f"Staying at difficulty level {current_difficulty}")
                # If we're at max difficulty or not moving up, we're done
                break
        
        return result

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
        self.current_difficulty = 0
        self.max_difficulty = len(self.difficulty_mapping) - 1
        self.current_indices = self._get_indices_for_difficulty(self.current_difficulty)
    
    def _difficulty_mapping(self, difficulty):
        """Default mapping of difficulty to dataset indices."""
        # Example: difficulties 0-4, each difficulty adds 20% more of the dataset
        dataset_size = len(self.base_dataset)
        num_examples = int(dataset_size * (difficulty + 1) * 0.2)
        return list(range(min(num_examples, dataset_size)))
    
    def _get_indices_for_difficulty(self, difficulty):
        """Get indices for the current difficulty level."""
        if callable(self.difficulty_mapping):
            return self.difficulty_mapping(difficulty)
        else:
            return self.difficulty_mapping.get(difficulty, [])
    
    def set_difficulty(self, difficulty):
        """Set the current difficulty level."""
        self.current_difficulty = min(difficulty, self.max_difficulty)
        self.current_indices = self._get_indices_for_difficulty(self.current_difficulty)
    
    def __len__(self):
        return len(self.current_indices)
    
    def __getitem__(self, idx):
        if idx >= len(self.current_indices):
            raise IndexError(f"Index {idx} out of range for difficulty {self.current_difficulty}")
        base_idx = self.current_indices[idx]
        return self.base_dataset[base_idx]