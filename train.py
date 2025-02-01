import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from model import FruitDetectionModel, get_transform
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import sys

class TrainingManager:
    def __init__(self):
        # Check GPU availability first
        if not torch.cuda.is_available():
            print("Error: GPU is not available!")
            print("This script requires GPU for training.")
            print("\nPlease:")
            print("1. Check your NVIDIA drivers")
            print("2. Ensure CUDA is properly installed")
            print("3. Verify PyTorch CUDA installation")
            sys.exit(1)
            
        self.device = torch.device('cuda')
        print(f"\nGPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        
        self.best_accuracy = 0
        self.train_losses = []
        self.val_accuracies = []

    def train_model(self, num_epochs=20):
        print("\nInitializing training with GPU...")
        print(f"Initial GPU Memory allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
        
        # Data loading with GPU pinned memory
        train_transform = get_transform(is_training=True)
        val_transform = get_transform(is_training=False)
        
        train_dataset = datasets.ImageFolder('dataset/train', transform=train_transform)
        val_dataset = datasets.ImageFolder('dataset/test', transform=val_transform)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Model initialization
        model = FruitDetectionModel(num_classes=len(train_dataset.classes)).to(self.device)
        
        # Enable parallel GPU training if multiple GPUs available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
            
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), 
                              lr=0.002,
                              weight_decay=0.01)
        
        scheduler = optim.lr_scheduler.LinearLR(optimizer,
                                              start_factor=1.0,
                                              end_factor=0.01,
                                              total_iters=20)
        
        print(f"\nStarting training with {len(train_dataset)} images")
        print(f"Validating with {len(val_dataset)} images")
        print(f"Training for {num_epochs} epochs")
        
        try:
            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                
                for i, (images, labels) in enumerate(train_loader):
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    if i % 10 == 9:
                        avg_loss = running_loss/10
                        self.train_losses.append(avg_loss)
                        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], '
                              f'Loss: {avg_loss:.4f}, '
                              f'GPU Memory: {torch.cuda.memory_allocated(0)/1e9:.2f} GB')
                        running_loss = 0.0
                
                # Validation phase
                model.eval()
                correct = 0
                total = 0
                all_preds = []
                all_labels = []
                
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                
                accuracy = 100 * correct / total
                self.val_accuracies.append(accuracy)
                
                print(f'Epoch [{epoch+1}/{num_epochs}] Validation Accuracy: {accuracy:.2f}%')
                
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_state,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'accuracy': accuracy,
                        'classes': train_dataset.classes
                    }, 'best_model.pth')
                    
                    # Plot confusion matrix
                    cm = confusion_matrix(all_labels, all_preds)
                    plt.figure(figsize=(15, 10))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                              xticklabels=train_dataset.classes,
                              yticklabels=train_dataset.classes)
                    plt.title('Confusion Matrix')
                    plt.ylabel('True Label')
                    plt.xlabel('Predicted Label')
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig('static/confusion_matrix.png')
                    plt.close()
                
                scheduler.step()
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user!")
        except Exception as e:
            print(f"\nError during training: {str(e)}")
        finally:
            # Save training plots
            plt.figure(figsize=(10, 5))
            plt.plot(self.train_losses)
            plt.title('Training Loss Over Time')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.savefig('static/training_loss.png')
            plt.close()
            
            plt.figure(figsize=(10, 5))
            plt.plot(self.val_accuracies)
            plt.title('Validation Accuracy Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.savefig('static/validation_accuracy.png')
            plt.close()
            
            print(f"\nTraining completed! Best accuracy: {self.best_accuracy:.2f}%")
            print(f"Final GPU Memory: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

if __name__ == '__main__':
    print("\nChecking GPU availability...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        trainer = TrainingManager()
        trainer.train_model()
    else:
        print("\nError: GPU is not available!")
        print("This script requires GPU for training.")
        print("\nPlease:")
        print("1. Check your NVIDIA drivers")
        print("2. Ensure CUDA is properly installed")
        print("3. Verify PyTorch CUDA installation")
        sys.exit(1) 