# testing.py
from utils.metrics import psnr
import torch

def test(model, device, test_loader, l1_criterion, mse_criterion):
    model.eval()  # Set the model to evaluation mode
    test_running_loss = 0.0
    test_running_psnr = 0.0
    with torch.no_grad():  # No gradients needed for evaluation
        for before_image, after_image in test_loader:
            before_image, after_image = before_image.to(device), after_image.to(device)
            outputs = model(before_image)
            
            # Calculate loss
            l1_loss = l1_criterion(outputs, after_image)
            mse_loss = mse_criterion(outputs, after_image)
            loss = l1_loss + mse_loss
            
            # Update running loss
            test_running_loss += loss.item()
            
            # Calculate and update PSNR
            batch_psnr = psnr(outputs, after_image, max_pixel=1.0)  # Adjust max_pixel if necessary
            test_running_psnr += batch_psnr
        
        # Calculate average loss and PSNR over the test dataset
        test_loss = test_running_loss / len(test_loader)
        test_psnr = test_running_psnr / len(test_loader)
    
    return test_loss, test_psnr
