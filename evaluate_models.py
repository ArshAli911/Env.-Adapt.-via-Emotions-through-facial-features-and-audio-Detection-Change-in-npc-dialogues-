"""
Model Evaluation Script for VR Emotion Adaptation System
Evaluates accuracy of facial, audio, and multimodal emotion recognition models
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
import pandas as pd
import time
from tqdm import tqdm

# Import your existing modules
from models.emotion_models import FacialEmotionCNN, AudioEmotionLSTM, EmotionClassifier
from data.data_loader import FacialExpressionDataset, AudioEmotionDataset
from main import VREmotionAdaptation

class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, model_type='facial', device='auto'):
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)
        print(f"Evaluating on device: {self.device}")
        
        # Emotion labels
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Initialize model
        self.classifier = EmotionClassifier(model_type, str(self.device))
        
        # Evaluation results
        self.results = {
            'predictions': [],
            'true_labels': [],
            'probabilities': [],
            'confidences': [],
            'inference_times': []
        }
    
    def load_test_data(self, data_path):
        """Load test dataset"""
        print(f"Loading test data from: {data_path}")
        
        if self.model_type == 'facial':
            dataset = FacialExpressionDataset(data_path)
        elif self.model_type == 'audio':
            dataset = AudioEmotionDataset(data_path)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return dataset
    
    def evaluate_model(self, test_dataset, batch_size=1):
        """Evaluate model on test dataset"""
        print(f"Evaluating {self.model_type} model...")
        
        self.classifier.model.eval()
        total_samples = len(test_dataset)
        
        with torch.no_grad():
            for i in tqdm(range(total_samples), desc="Evaluating"):
                # Get single sample
                data, label = test_dataset[i]
                
                # Convert to tensor
                if self.model_type == 'facial':
                    input_tensor = data.unsqueeze(0).to(self.device)  # Add batch dimension
                else:  # audio
                    input_tensor = data.unsqueeze(0).to(self.device)  # Add batch dimension
                
                # Measure inference time
                start_time = time.time()
                result = self.classifier.predict(input_tensor)
                inference_time = time.time() - start_time
                
                # Store results
                self.results['predictions'].append(result['emotion'])
                self.results['true_labels'].append(label)
                self.results['probabilities'].append(result['probabilities'].flatten())
                self.results['confidences'].append(result['confidence'])
                self.results['inference_times'].append(inference_time)
    
    def calculate_metrics(self):
        """Calculate comprehensive accuracy metrics"""
        print("Calculating accuracy metrics...")
        
        # Convert predictions to numerical labels if they're strings
        predictions = []
        for pred in self.results['predictions']:
            if isinstance(pred, str):
                # Convert string emotion to numerical label
                pred_idx = self.emotion_labels.index(pred)
                predictions.append(pred_idx)
            else:
                predictions.append(pred)
        
        # Convert true labels to numerical if they're tensors
        true_labels = []
        for label in self.results['true_labels']:
            if isinstance(label, torch.Tensor):
                true_labels.append(label.item())
            else:
                true_labels.append(label)
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        confidences = np.array(self.results['confidences'])
        inference_times = np.array(self.results['inference_times'])
        
        # Basic accuracy
        accuracy = accuracy_score(true_labels, predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average=None, labels=range(len(self.emotion_labels))
        )
        
        # Macro and weighted averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='macro'
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        # Performance metrics
        avg_inference_time = np.mean(inference_times)
        avg_confidence = np.mean(confidences)
        
        # Create results dictionary
        metrics = {
            'overall_accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'per_class_precision': precision,
            'per_class_recall': recall,
            'per_class_f1': f1,
            'per_class_support': support,
            'avg_inference_time': avg_inference_time,
            'avg_confidence': avg_confidence,
            'total_samples': len(predictions)
        }
        
        return metrics
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot confusion matrix"""
        # Convert predictions and true_labels to integer indices if needed
        predictions = [self.emotion_labels.index(p) if isinstance(p, str) else p for p in self.results['predictions']]
        true_labels = [self.emotion_labels.index(t) if isinstance(t, str) else (t.item() if hasattr(t, 'item') else t) for t in self.results['true_labels']]
        
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.emotion_labels, 
                   yticklabels=self.emotion_labels)
        plt.title(f'Confusion Matrix - {self.model_type.capitalize()} Model')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        plt.show()
    
    def plot_accuracy_by_emotion(self, metrics, save_path=None):
        """Plot accuracy metrics by emotion"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Precision by emotion
        ax1.bar(self.emotion_labels, metrics['per_class_precision'])
        ax1.set_title('Precision by Emotion')
        ax1.set_ylabel('Precision')
        ax1.tick_params(axis='x', rotation=45)
        
        # Recall by emotion
        ax2.bar(self.emotion_labels, metrics['per_class_recall'])
        ax2.set_title('Recall by Emotion')
        ax2.set_ylabel('Recall')
        ax2.tick_params(axis='x', rotation=45)
        
        # F1 Score by emotion
        ax3.bar(self.emotion_labels, metrics['per_class_f1'])
        ax3.set_title('F1 Score by Emotion')
        ax3.set_ylabel('F1 Score')
        ax3.tick_params(axis='x', rotation=45)
        
        # Support (number of samples) by emotion
        ax4.bar(self.emotion_labels, metrics['per_class_support'])
        ax4.set_title('Number of Samples by Emotion')
        ax4.set_ylabel('Support')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Accuracy by emotion plot saved to {save_path}")
        plt.show()
    
    def plot_confidence_distribution(self, save_path=None):
        """Plot confidence distribution"""
        confidences = np.array(self.results['confidences'])
        
        plt.figure(figsize=(12, 5))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(confidences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidences):.3f}')
        plt.legend()
        
        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(confidences)
        plt.title('Confidence Box Plot')
        plt.ylabel('Confidence')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confidence distribution saved to {save_path}")
        plt.show()
    
    def print_detailed_report(self, metrics):
        """Print detailed evaluation report"""
        print("\n" + "="*60)
        print(f"EVALUATION REPORT - {self.model_type.upper()} MODEL")
        print("="*60)
        
        print(f"\n�� OVERALL METRICS:")
        print(f"   Overall Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)")
        print(f"   Macro Precision:  {metrics['macro_precision']:.4f}")
        print(f"   Macro Recall:     {metrics['macro_recall']:.4f}")
        print(f"   Macro F1-Score:   {metrics['macro_f1']:.4f}")
        print(f"   Weighted Precision: {metrics['weighted_precision']:.4f}")
        print(f"   Weighted Recall:    {metrics['weighted_recall']:.4f}")
        print(f"   Weighted F1-Score:  {metrics['weighted_f1']:.4f}")
        
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"   Average Inference Time: {metrics['avg_inference_time']*1000:.2f} ms")
        print(f"   Average Confidence:     {metrics['avg_confidence']:.4f}")
        print(f"   Total Samples:          {metrics['total_samples']}")
        
        print(f"\n📈 PER-EMOTION METRICS:")
        print(f"{'Emotion':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 55)
        for i, emotion in enumerate(self.emotion_labels):
            print(f"{emotion:<12} {metrics['per_class_precision'][i]:<10.4f} "
                  f"{metrics['per_class_recall'][i]:<10.4f} "
                  f"{metrics['per_class_f1'][i]:<10.4f} "
                  f"{metrics['per_class_support'][i]:<10}")
        
        # Performance assessment
        print(f"\n🎯 PERFORMANCE ASSESSMENT:")
        if metrics['overall_accuracy'] >= 0.9:
            print("   🟢 EXCELLENT: Very high accuracy")
        elif metrics['overall_accuracy'] >= 0.8:
            print("   🟡 GOOD: High accuracy")
        elif metrics['overall_accuracy'] >= 0.7:
            print("   🟠 FAIR: Moderate accuracy")
        else:
            print("   🔴 POOR: Low accuracy - needs improvement")
        
        if metrics['avg_inference_time'] < 0.1:
            print("   ⚡ FAST: Real-time performance achieved")
        elif metrics['avg_inference_time'] < 0.5:
            print("   🐌 MODERATE: Acceptable performance")
        else:
            print("   🐌 SLOW: Performance needs optimization")
    
    def save_results(self, metrics, output_dir='evaluation_results'):
        """Save evaluation results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'Metric': ['Overall Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1',
                      'Weighted Precision', 'Weighted Recall', 'Weighted F1',
                      'Avg Inference Time (ms)', 'Avg Confidence', 'Total Samples'],
            'Value': [metrics['overall_accuracy'], metrics['macro_precision'], 
                     metrics['macro_recall'], metrics['macro_f1'],
                     metrics['weighted_precision'], metrics['weighted_recall'], 
                     metrics['weighted_f1'], metrics['avg_inference_time']*1000,
                     metrics['avg_confidence'], metrics['total_samples']]
        })
        
        csv_path = os.path.join(output_dir, f'{self.model_type}_metrics.csv')
        metrics_df.to_csv(csv_path, index=False)
        print(f"Metrics saved to {csv_path}")
        
        # Save per-class metrics
        per_class_df = pd.DataFrame({
            'Emotion': self.emotion_labels,
            'Precision': metrics['per_class_precision'],
            'Recall': metrics['per_class_recall'],
            'F1_Score': metrics['per_class_f1'],
            'Support': metrics['per_class_support']
        })
        
        per_class_path = os.path.join(output_dir, f'{self.model_type}_per_class_metrics.csv')
        per_class_df.to_csv(per_class_path, index=False)
        print(f"Per-class metrics saved to {per_class_path}")
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'True_Label': self.results['true_labels'],
            'Predicted_Label': self.results['predictions'],
            'Confidence': self.results['confidences'],
            'Inference_Time_ms': [t*1000 for t in self.results['inference_times']]
        })
        
        predictions_path = os.path.join(output_dir, f'{self.model_type}_predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Predictions saved to {predictions_path}")

def evaluate_all_models():
    """Evaluate all models (facial, audio, multimodal)"""
    print("🎭 VR Emotion Adaptation System - Model Evaluation")
    print("="*60)
    
    # Create output directory
    os.makedirs('evaluation_results', exist_ok=True)
    
    # Test data paths (adjust these to your actual test data paths)
    facial_test_path = "archive (1)/test"  # Facial emotion test data
    audio_test_path = "archive/audio_speech_actors_01-24"  # Audio emotion test data
    
    # Evaluate facial model
    if os.path.exists(facial_test_path):
        print("\n🔍 Evaluating Facial Emotion Model...")
        facial_evaluator = ModelEvaluator('facial')
        facial_dataset = facial_evaluator.load_test_data(facial_test_path)
        facial_evaluator.evaluate_model(facial_dataset)
        facial_metrics = facial_evaluator.calculate_metrics()
        facial_evaluator.print_detailed_report(facial_metrics)
        facial_evaluator.plot_confusion_matrix('evaluation_results/facial_confusion_matrix.png')
        facial_evaluator.plot_accuracy_by_emotion(facial_metrics, 'evaluation_results/facial_accuracy_by_emotion.png')
        facial_evaluator.plot_confidence_distribution('evaluation_results/facial_confidence_distribution.png')
        facial_evaluator.save_results(facial_metrics, 'evaluation_results')
    else:
        print(f"⚠️ Facial test data not found at {facial_test_path}")
    
    # Evaluate audio model
    if os.path.exists(audio_test_path):
        print("\n🎵 Evaluating Audio Emotion Model...")
        audio_evaluator = ModelEvaluator('audio')
        audio_dataset = audio_evaluator.load_test_data(audio_test_path)
        audio_evaluator.evaluate_model(audio_dataset)
        audio_metrics = audio_evaluator.calculate_metrics()
        audio_evaluator.print_detailed_report(audio_metrics)
        audio_evaluator.plot_confusion_matrix('evaluation_results/audio_confusion_matrix.png')
        audio_evaluator.plot_accuracy_by_emotion(audio_metrics, 'evaluation_results/audio_accuracy_by_emotion.png')
        audio_evaluator.plot_confidence_distribution('evaluation_results/audio_confidence_distribution.png')
        audio_evaluator.save_results(audio_metrics, 'evaluation_results')
    else:
        print(f"⚠️ Audio test data not found at {audio_test_path}")
    
    print("\n✅ Model evaluation completed!")
    print("📁 Results saved in 'evaluation_results' directory")

if __name__ == "__main__":
    evaluate_all_models() 