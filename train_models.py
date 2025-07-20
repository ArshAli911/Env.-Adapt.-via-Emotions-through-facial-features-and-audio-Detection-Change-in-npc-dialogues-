"""
Training script for emotion recognition models
Trains facial, audio, and multimodal emotion recognition models
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from data.data_loader import FacialExpressionDataset, AudioEmotionDataset, create_data_loaders
from models.emotion_models import FacialEmotionCNN, AudioEmotionLSTM, MultiModalEmotionFusion
from config import config

class ModelTrainer:
    """Trainer class for emotion recognition models"""
    
    def __init__(self, model_type='facial', device='auto'):
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)
        print(f"Training on device: {self.device}")
        
        # Initialize model
        if model_type == 'facial':
            self.model = FacialEmotionCNN().to(self.device)
        elif model_type == 'audio':
            self.model = AudioEmotionLSTM().to(self.device)
        elif model_type == 'multimodal':
            self.model = MultiModalEmotionFusion().to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Training parameters
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if isinstance(data, (list, tuple)):
                # Multi-modal data
                data = [d.to(self.device) for d in data]
            else:
                data = data.to(self.device)
            target = target.squeeze().to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(*data) if isinstance(data, (list, tuple)) else self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        return total_loss / len(train_loader), correct / total
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                if isinstance(data, (list, tuple)):
                    data = [d.to(self.device) for d in data]
                else:
                    data = data.to(self.device)
                target = target.squeeze().to(self.device)
                
                output = self.model(*data) if isinstance(data, (list, tuple)) else self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        return total_loss / len(val_loader), correct / total, all_predictions, all_targets
    
    def train(self, train_loader, val_loader, epochs=50, save_path=None):
        """Train the model"""
        best_val_acc = 0
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation
            val_loss, val_acc, predictions, targets = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Save best model
            if val_acc > best_val_acc and save_path:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print(f'New best model saved with validation accuracy: {val_acc:.4f}')
        
        # Final evaluation
        print(f'\nTraining completed. Best validation accuracy: {best_val_acc:.4f}')
        return predictions, targets
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f'Training history saved to {save_path}')
        plt.show()
    
    def plot_confusion_matrix(self, predictions, targets, save_path=None):
        """Plot confusion matrix"""
        emotion_labels = config.model.emotion_labels
        
        cm = confusion_matrix(targets, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=emotion_labels, yticklabels=emotion_labels)
        plt.title(f'Confusion Matrix - {self.model_type.capitalize()} Model')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path)
            print(f'Confusion matrix saved to {save_path}')
        plt.show()

def main():
    """Main training function"""
    # Create output directory
    os.makedirs('models', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # Train facial emotion model
    print("Training Facial Emotion Recognition Model...")
    facial_trainer = ModelTrainer('facial')
    
    # Load facial data
    facial_dataset = FacialExpressionDataset(config.data.facial_data_path)
    facial_train_loader = DataLoader(facial_dataset, batch_size=32, shuffle=True)
    facial_val_loader = DataLoader(facial_dataset, batch_size=32, shuffle=False)
    
    # Train facial model
    predictions, targets = facial_trainer.train(
        facial_train_loader, 
        facial_val_loader, 
        epochs=30,
        save_path=config.model.facial_model_weights
    )
    
    # Plot results
    facial_trainer.plot_training_history('output/facial_training_history.png')
    facial_trainer.plot_confusion_matrix(predictions, targets, 'output/facial_confusion_matrix.png')
    
    # Train audio emotion model
    print("\nTraining Audio Emotion Recognition Model...")
    audio_trainer = ModelTrainer('audio')
    
    # Load audio data
    audio_dataset = AudioEmotionDataset(config.data.audio_data_path)
    audio_train_loader = DataLoader(audio_dataset, batch_size=32, shuffle=True)
    audio_val_loader = DataLoader(audio_dataset, batch_size=32, shuffle=False)
    
    # Train audio model
    predictions, targets = audio_trainer.train(
        audio_train_loader, 
        audio_val_loader, 
        epochs=32,
        save_path=config.model.audio_model_weights
    )
    
    # Plot results
    audio_trainer.plot_training_history('output/audio_training_history.png')
    audio_trainer.plot_confusion_matrix(predictions, targets, 'output/audio_confusion_matrix.png')
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main() 