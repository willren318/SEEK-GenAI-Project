from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def calculate_metrics(y_true, y_pred):
    """Calculate performance metrics"""
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    metrics = {
        'accuracy': acc,
        'report': report,
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    return metrics

def print_metrics(metrics, report_path=None):
    """Print evaluation metrics in a readable format"""
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nClassification Report:")
    # Convert dict report back to string format
    report_df = pd.DataFrame(metrics['report']).transpose()
    print(report_df.round(3))
    # Save classification report to file
    with open(report_path, 'w') as f:
        f.write(report_df.to_string())
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])

def plot_confusion_matrix(metrics, labels=None, save_path=None):
    """Plot confusion matrix as a heatmap"""
    cm = metrics['confusion_matrix']
    plt.figure(figsize=(10, 8))
    
    if labels is None:
        # Try to extract labels from the report
        try:
            labels = list(metrics['report'].keys())
            labels = [l for l in labels if l not in ['accuracy', 'macro avg', 'weighted avg']]
        except:
            labels = list(range(cm.shape[0]))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        plt.close()  # Close the figure to prevent display
    else:
        # Only show if not saving to file
        plt.show()

def save_predictions(df, save_path):
    """Save predictions to CSV file"""
    df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")
