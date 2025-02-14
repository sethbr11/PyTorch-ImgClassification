import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F

from config import get_class_names

def validate(model, test_loader, device, epoch, validate_with_images=False):
    class_names = get_class_names()
    selected_images, predicted, actual, final_loss = run_validation(model, test_loader, device)
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

def run_validation(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0

    # Randomly select two images from the test_loader
    all_images, all_labels = next(iter(test_loader))
    indices = random.sample(range(len(all_images)), 2)
    selected_images = all_images[indices].to(device)
    selected_labels = all_labels[indices].to(device)

    with torch.no_grad():
        outputs = model(selected_images)
        loss = criterion(outputs, selected_labels)
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)

    avg_loss = total_loss
    return selected_images.cpu(), predicted.cpu().numpy(), selected_labels.cpu().numpy(), avg_loss