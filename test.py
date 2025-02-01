import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from model import FruitDetectionModel, get_transform
import numpy as np
import os

def get_num_classes():
    train_path = 'dataset/train'
    return len([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])

def plot_detailed_confusion_matrix(y_true, y_pred, classes):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure and axes
    plt.figure(figsize=(15, 10))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    
    # Customize the plot
    plt.title('Confusion Matrix', pad=20, size=15)
    plt.ylabel('True Label', size=12)
    plt.xlabel('Predicted Label', size=12)
    
    # Rotate tick labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved as 'confusion_matrix.png'")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))
    
    plt.close()

def test_model():
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get number of classes from training set
    num_classes = get_num_classes()
    print(f"Number of classes: {num_classes}")
    
    # Load test dataset
    transform = get_transform()
    try:
        test_dataset = datasets.ImageFolder(
            'dataset/test',
            transform=transform
        )
        print(f"\nFound {len(test_dataset)} test images")
        print("Test categories:", test_dataset.classes)
    except Exception as e:
        print(f"Error loading test dataset: {str(e)}")
        return
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model and load weights
    model = FruitDetectionModel(num_classes=num_classes).to(device)
    
    try:
        # Load the model with CPU mapping if CUDA is not available
        model.load_state_dict(
            torch.load('fruit_model.pth', 
                      map_location=torch.device('cpu'))
        )
        print("Model weights loaded successfully!")
    except FileNotFoundError:
        print("Error: Could not find 'fruit_model.pth'. Please make sure the model is trained first.")
        return
    except Exception as e:
        print(f"Error loading model weights: {str(e)}")
        return
    
    # Evaluate
    model.eval()
    all_preds = []
    all_labels = []
    
    print("\nStarting evaluation...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Get class names
    classes = test_dataset.classes
    
    # Plot confusion matrix and print classification report
    plot_detailed_confusion_matrix(all_labels, all_preds, classes)
    
    # Calculate and print overall accuracy
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    print(f'\nOverall Test Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    test_model() 