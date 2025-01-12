class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, mode='min'):
        """
        Args:
            patience (int): How many epochs to wait before stopping when loss is
                          not improving
            min_delta (float): Minimum change in the monitored quantity to qualify 
                             as an improvement
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop 
                       when quantity monitored has stopped decreasing; in 'max' mode 
                       it will stop when the quantity monitored has stopped increasing
        """
        self.patience = patience
        self.min_delta = min_delta
        if mode not in ['min', 'max']:
            raise ValueError(f"Early stopping mode must be 'min' or 'max', got {mode}")
        self.mode = mode
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.min_delta *= 1 if mode == 'min' else -1
    
    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
            return False
        
        if self.mode == 'min':
            delta = current_loss - self.best_loss
        else:
            delta = self.best_loss - current_loss
            
        if delta > self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = current_loss
            self.counter = 0
            
        return self.early_stop