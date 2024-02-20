import torch.optim as optim

def setup_optimizer_scheduler(model, config):
    # Extract optimizer parameters from config
    optimizer_params = config['optimizer']['params']
    optimizer_type = config['optimizer']['type'].lower()
    
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), **optimizer_params)
    # Add more conditions here if you plan to support more optimizers
    
    # Extract scheduler parameters from config
    scheduler_params = config['scheduler']['params']
    scheduler_type = config['scheduler']['type']
    
    if scheduler_type == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
    # Add more conditions here if you plan to support more schedulers
    
    return optimizer, scheduler
