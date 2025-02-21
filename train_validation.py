import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F

from config import get_class_names
            
def validate(model, val_loader, device, epoch, validate_with_images=False):
    class_names = get_class_names()
    selected_images, predicted, actual, final_loss = run_validation(model, val_loader, device)
    predicted_classes = [class_names[p] for p in predicted]
    actual_classes = [class_names[a] for a in actual]
    print(f"Validation - Predicted: {predicted_classes}, Actual: {actual_classes}, Loss: {final_loss:.4f}")
    
    # Display the images if the flag is set
    if validate_with_images:
        if not plt.get_backend().lower().startswith('agg'):
            fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=150)  # Small size to match image resolution
            fig.canvas.manager.set_window_title(f'Validation Results - Epoch {epoch}')  # Set the window title
            for i, ax in enumerate(axes):
                # Denormalize and permute for display
                img = selected_images[i] * 0.5 + 0.5  # Assuming normalization was (0.5, 0.5)
                
                # Optional: Upscale for better visibility (comment this out if unnecessary)
                img = F.interpolate(img.unsqueeze(0), size=(128, 128), mode='nearest').squeeze(0)

                ax.imshow(img.permute(1, 2, 0).cpu().numpy(), interpolation="nearest")  # Nearest preserves sharpness
                ax.set_title(f"Pred: {predicted_classes[i]}\nActual: {actual_classes[i]}")
                ax.axis('off')
            
            plt.show()
        else:
            print("No display found. Skipping image display.")
    
    return final_loss

def run_validation(model, val_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    all_images = []
    all_labels = []

    # Iterate through the entire validation set
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Collect all images and labels for loss calculation
            all_images.append(images)
            all_labels.append(labels)

    # Flatten the lists to obtain all images and labels
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Randomly select two images from the validation set
    indices = random.sample(range(len(all_images)), 2)
    selected_images = all_images[indices].to(device)
    selected_labels = all_labels[indices].to(device)

    # Get the predictions for the selected images
    with torch.no_grad():
        outputs = model(selected_images)
        _, predicted = torch.max(outputs, 1)

    # Calculate the average loss
    avg_loss = total_loss / len(val_loader)  # Average over all batches in the val_loader

    return selected_images.cpu(), predicted.cpu().numpy(), selected_labels.cpu().numpy(), avg_loss