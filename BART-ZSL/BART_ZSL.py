# ==================== Import Necessary Libraries ====================
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical computations
from sklearn.model_selection import StratifiedKFold, train_test_split  # For creating stratified folds and train/test splits
from sklearn.metrics import (  # For calculating various evaluation metrics
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    classification_report,
    confusion_matrix
)
from transformers import pipeline  # Hugging Face pipeline for easy model usage
import matplotlib.pyplot as plt  # For plotting graphs and charts
import seaborn as sns  # For advanced visualization (e.g., heatmaps)
import torch  # PyTorch for GPU acceleration if available
import pickle  # For saving/loading Python objects (e.g., fold indices)
import os  # For checking file/folder existence
from scipy.stats import ttest_rel  # For statistical tests between paired samples (optional)

# ==================== Seed Setting for Reproducibility ====================
SEED = 42
np.random.seed(SEED)  # Set NumPy seed
torch.manual_seed(SEED)  # Set PyTorch seed
# ====================== End of Seed Setting Section ========================

# ==================== Configuration Parameters ====================
DATA_PATH = r'C:\Dataset\preprocessed_combined_depression_dataset_good.csv'  # Path to the dataset file
TEST_SIZE = 0.2  # Percentage of data to use for the test set
K_FOLDS = 5  # Number of folds for cross-validation
CANDIDATE_LABELS = ['Not Depressed', 'Depressed']  # Labels for zero-shot classification
NEW_TEXTS = [  # New texts to classify after training
    "I'm depressed.",
    "It's a bright sunny day!",
    "I don't know how to cope anymore.",
    "Looking forward to the weekend."
]
FOLD_INDICES_FILE = 'fold_indices.pkl'  # Filename to save/load stratified fold indices
# ====================== End of Configuration ============================

# ==================== Step 1: Load and Prepare the Dataset ====================
def load_and_prepare_data(data_path):
    """
    Loads the dataset, checks for required columns, removes missing/empty entries,
    and maps labels to integers if necessary.
    """
    df = pd.read_csv(data_path)  # Load CSV into a pandas DataFrame
    
    required_columns = ['message to examine', 'label']
    for col in required_columns:
        # Ensure the required columns are present
        if col not in df.columns:
            raise ValueError(f"Dataset must contain '{col}' column.")
    
    # Drop rows with missing values in the required columns
    df = df.dropna(subset=required_columns)
    
    # Convert text column to string and remove empty texts
    df['message to examine'] = df['message to examine'].astype(str)
    df = df[df['message to examine'].str.strip() != '']
    
    # If labels are strings, map them to integers
    if df['label'].dtype == 'object':
        label_mapping = {'Not Depressed': 0, 'Depressed': 1}
        df['label'] = df['label'].map(label_mapping)
    
    # Ensure label column is of integer type
    df['label'] = df['label'].astype(int)
    
    return df

# Load and prepare the dataset
df = load_and_prepare_data(DATA_PATH)
print(f"Total samples after preprocessing: {len(df)}")

# ==================== Step 2: Split into Training+Validation and Test Sets ====================
def split_data(df, test_size=0.2, seed=SEED):
    """
    Splits the data into training+validation and test sets, using stratification to maintain label balance.
    """
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        df['message to examine'].tolist(),
        df['label'].tolist(),
        test_size=test_size,
        random_state=seed,
        stratify=df['label']
    )
    
    print(f"Training + Validation samples: {len(train_val_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    return train_val_texts, test_texts, train_val_labels, test_labels

# Perform the split
train_val_texts, test_texts, train_val_labels, test_labels = split_data(df, TEST_SIZE, SEED)

# ==================== Step 3: Initialize the BART-ZSL Pipeline ====================
def initialize_bart_zsl_pipeline():
    """
    Initializes the zero-shot classification pipeline with a BART-based model (facebook/bart-large-mnli).
    Automatically uses GPU if available.
    """
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, else CPU
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device,
        batch_size=16  # Set batch size for inference
    )
    return classifier

# Initialize the classifier pipeline
classifier = initialize_bart_zsl_pipeline()
print("BART-ZSL pipeline initialized.")

# ==================== Step 4: Create or Load Fold Indices ====================
def create_fold_indices(texts, labels, k_folds=5, seed=SEED):
    """
    Creates stratified k-fold splits for cross-validation and returns their indices.
    """
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    fold_indices = []
    # Generate train/validation indices for each fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        fold_indices.append((train_idx, val_idx))
        print(f"Fold {fold + 1}: Train indices={len(train_idx)}, Validation indices={len(val_idx)}")
    return fold_indices

def load_fold_indices(filename='fold_indices.pkl'):
    """
    Loads precomputed fold indices from a pickle file.
    """
    with open(filename, 'rb') as f:
        fold_indices = pickle.load(f)
    print(f"Fold indices loaded from '{filename}'")
    return fold_indices

def save_fold_indices(fold_indices, filename='fold_indices.pkl'):
    """
    Saves fold indices to a pickle file.
    """
    with open(filename, 'wb') as f:
        pickle.dump(fold_indices, f)
    print(f"Fold indices saved to '{filename}'")

# Check if fold indices exist; if not, create and save them
if os.path.exists(FOLD_INDICES_FILE):
    fold_indices = load_fold_indices(FOLD_INDICES_FILE)
else:
    fold_indices = create_fold_indices(train_val_texts, train_val_labels, K_FOLDS, SEED)
    save_fold_indices(fold_indices, FOLD_INDICES_FILE)

# ==================== Step 5: Define the Evaluation Function ====================
def evaluate_bart_zsl_model(test_texts, test_labels, classifier, candidate_labels):
    """
    Evaluates the BART-ZSL model on given test texts and labels.
    Returns a dictionary of performance metrics.
    """
    # Perform zero-shot classification on test texts
    results = classifier(test_texts, candidate_labels, multi_label=False)
    
    # Extract top predicted label and its confidence for each sample
    predicted_labels = [result['labels'][0] for result in results]
    confidence_scores = [result['scores'][0] for result in results]
    
    # Map predicted labels (strings) to binary values (0 or 1)
    binary_label_mapping = {'Depressed': 1, 'Not Depressed': 0}
    y_pred_binary = [binary_label_mapping[label] for label in predicted_labels]
    
    # Ground truth labels are already integers (0 or 1)
    y_true_binary = test_labels
    
    # Extract scores for the 'Depressed' class for ROC calculations
    y_scores = [score if label == 'Depressed' else 0 for score, label in zip(confidence_scores, predicted_labels)]
    
    # Compute evaluation metrics
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    # Compute AUC-ROC, handling cases where only one class might be present
    try:
        roc_auc = roc_auc_score(y_true_binary, y_scores)
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_scores)
    except ValueError:
        roc_auc = float('nan')
        fpr, tpr, thresholds = [0], [0], [0]
    
    # Generate a classification report and confusion matrix
    report = classification_report(y_true_binary, y_pred_binary, target_names=['Not Depressed', 'Depressed'], zero_division=0)
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc,
        'classification_report': report,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr
    }

# ==================== Step 6: Evaluate BART-ZSL Across All Folds ====================
def evaluate_across_folds(fold_indices, train_val_texts, train_val_labels, classifier, candidate_labels):
    """
    Performs evaluation of the BART-ZSL model across all folds and aggregates the metrics.
    """
    # Dictionary to store metrics for each fold
    metrics_bart = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': [],
        'classification_report': [],
        'confusion_matrix': [],
        'fpr': [],
        'tpr': []
    }
    
    # Evaluate on each fold
    for fold, (train_idx, val_idx) in enumerate(fold_indices):
        print(f"\nEvaluating Fold {fold + 1}/{K_FOLDS}")
        
        # Validation fold serves as the "test set" here
        test_texts_fold = [train_val_texts[i] for i in val_idx]
        test_labels_fold = [train_val_labels[i] for i in val_idx]
        
        # Evaluate model on this fold's validation set
        bart_metrics = evaluate_bart_zsl_model(test_texts_fold, test_labels_fold, classifier, candidate_labels)
        
        # Store metrics
        metrics_bart['accuracy'].append(bart_metrics['accuracy'])
        metrics_bart['precision'].append(bart_metrics['precision'])
        metrics_bart['recall'].append(bart_metrics['recall'])
        metrics_bart['f1'].append(bart_metrics['f1'])
        metrics_bart['auc'].append(bart_metrics['auc'])
        metrics_bart['classification_report'].append(bart_metrics['classification_report'])
        metrics_bart['confusion_matrix'].append(bart_metrics['confusion_matrix'])
        metrics_bart['fpr'].append(bart_metrics['fpr'])
        metrics_bart['tpr'].append(bart_metrics['tpr'])
        
        # Print metrics for this fold
        print(f"Fold {fold + 1} Metrics:")
        print(f"Accuracy: {bart_metrics['accuracy']:.4f}")
        print(f"Precision: {bart_metrics['precision']:.4f}")
        print(f"Recall: {bart_metrics['recall']:.4f}")
        print(f"F1-Score: {bart_metrics['f1']:.4f}")
        print(f"AUC-ROC: {bart_metrics['auc']:.4f}")
        print(f"Classification Report:\n{bart_metrics['classification_report']}")
    
    return metrics_bart

# Run evaluation across all folds
metrics_bart = evaluate_across_folds(fold_indices, train_val_texts, train_val_labels, classifier, CANDIDATE_LABELS)

# ==================== Step 7: Aggregate and Visualize Metrics ====================
def aggregate_and_visualize_metrics(metrics_bart):
    """
    Aggregates metrics across folds, prints mean/std, and creates a bar chart with error bars.
    """
    # Compute mean and standard deviation for each metric across folds
    bart_mean = (
        pd.Series(metrics_bart['accuracy']).mean(),
        pd.Series(metrics_bart['precision']).mean(),
        pd.Series(metrics_bart['recall']).mean(),
        pd.Series(metrics_bart['f1']).mean(),
        pd.Series(metrics_bart['auc']).mean()
    )
    
    bart_std = (
        pd.Series(metrics_bart['accuracy']).std(),
        pd.Series(metrics_bart['precision']).std(),
        pd.Series(metrics_bart['recall']).std(),
        pd.Series(metrics_bart['f1']).std(),
        pd.Series(metrics_bart['auc']).std()
    )
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    bart_means = bart_mean
    bart_stds = bart_std
    
    # Create a DataFrame for aggregated metrics
    aggregated_metrics = pd.DataFrame({
        'Metric': metrics,
        'Mean': bart_means,
        'Std Dev': bart_stds
    })
    
    # Print aggregated metrics
    print("\n========== BART-ZSL Aggregated Performance Across Folds ==========")
    print(aggregated_metrics.to_string(index=False))
    
    # Save aggregated metrics to CSV
    aggregated_metrics.to_csv('bart_zsl_aggregated_metrics.csv', index=False)
    print("Aggregated metrics saved to 'bart_zsl_aggregated_metrics.csv'")
    
    # Plot a bar chart with error bars
    x = np.arange(len(metrics))  # Positions for bars
    width = 0.6  # Bar width
    
    fig, ax = plt.subplots(figsize=(12, 7))
    rects = ax.bar(x, bart_means, width, yerr=bart_stds, label='BART-ZSL', capsize=5, color='salmon')
    
    # Labeling the chart
    ax.set_ylabel('Scores')
    ax.set_title('BART-ZSL Model Performance Across Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Add value labels on top of each bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects)
    
    plt.ylim(0, 1.0)  # Scores are between 0 and 1
    plt.tight_layout()
    plt.show()
    
    return {
        'metrics': metrics,
        'bart_means': bart_means,
        'bart_stds': bart_stds
    }

# Aggregate and visualize metrics across folds
aggregated_metrics = aggregate_and_visualize_metrics(metrics_bart)

# ==================== Step 8: Plot ROC Curves for Each Fold ====================
def plot_roc_curves(metrics_bart, k_folds=5):
    """
    Plots ROC curves for each fold to visualize model performance per fold.
    """
    plt.figure(figsize=(10, 8))
    
    # Plot each fold's ROC curve
    for fold in range(k_folds):
        fpr = metrics_bart['fpr'][fold]
        tpr = metrics_bart['tpr'][fold]
        roc_auc = metrics_bart['auc'][fold]
        plt.plot(fpr, tpr, lw=2, label=f'Fold {fold + 1} (AUC = {roc_auc:.2f})')
    
    # Plot a diagonal line for random guessing reference
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curves for BART-ZSL Across Folds')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# Plot the ROC curves for each fold
plot_roc_curves(metrics_bart, K_FOLDS)

# ==================== Step 9: Classify New Text Examples ====================
def classify_new_texts(new_texts, classifier, candidate_labels):
    """
    Uses the BART-ZSL classifier to predict labels for new text samples.
    """
    results = classifier(new_texts, candidate_labels, multi_label=False)
    
    predictions = [result['labels'][0] for result in results]
    confidence_scores = [result['scores'][0] for result in results]
    
    # Print the classification results for each new text
    print("\n========== New Texts Classification ==========")
    for text, pred, score in zip(new_texts, predictions, confidence_scores):
        print(f"Text: \"{text}\"\nPrediction: {pred} (Confidence: {score:.4f})\n")

# Classify example new texts
classify_new_texts(NEW_TEXTS, classifier, CANDIDATE_LABELS)

# ==================== Step 10: Evaluate BART-ZSL on Final Test Set ====================
def evaluate_on_final_test_set(test_texts, test_labels, classifier, candidate_labels):
    """
    Evaluates the BART-ZSL model on the final test set and visualizes the results.
    """
    bart_test_metrics = evaluate_bart_zsl_model(test_texts, test_labels, classifier, candidate_labels)
    
    print("\n========== BART-ZSL Final Test Set Evaluation ==========")
    print(f"Accuracy: {bart_test_metrics['accuracy']:.4f}")
    print(f"Precision: {bart_test_metrics['precision']:.4f}")
    print(f"Recall: {bart_test_metrics['recall']:.4f}")
    print(f"F1-Score: {bart_test_metrics['f1']:.4f}")
    print(f"AUC-ROC: {bart_test_metrics['auc']:.4f}")
    print(f"Classification Report:\n{bart_test_metrics['classification_report']}")
    
    # Plot confusion matrix
    cm = bart_test_metrics['confusion_matrix']
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Depressed', 'Depressed'],
                yticklabels=['Not Depressed', 'Depressed'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix for BART-ZSL Final Test Set')
    plt.show()
    
    # Plot ROC curve
    fpr = bart_test_metrics['fpr']
    tpr = bart_test_metrics['tpr']
    roc_auc = bart_test_metrics['auc']
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'BART-ZSL ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) - BART-ZSL Final Test Set')
    plt.legend(loc="lower right")
    plt.show()

# Evaluate on the final test set
evaluate_on_final_test_set(test_texts, test_labels, classifier, CANDIDATE_LABELS)

# ==================== Step 11: Save Detailed Metrics Per Fold ====================
def save_fold_metrics(metrics_bart, filename='bart_zsl_fold_metrics.csv'):
    """
    Saves the per-fold metrics into a CSV file for record-keeping.
    """
    fold_data = []
    for i in range(len(metrics_bart['accuracy'])):
        fold_data.append({
            'Fold': i+1,
            'Accuracy': metrics_bart['accuracy'][i],
            'Precision': metrics_bart['precision'][i],
            'Recall': metrics_bart['recall'][i],
            'F1-Score': metrics_bart['f1'][i],
            'AUC-ROC': metrics_bart['auc'][i]
        })
    
    # Convert to DataFrame and save as CSV
    df_metrics = pd.DataFrame(fold_data)
    df_metrics.to_csv(filename, index=False)
    print(f"Detailed fold metrics saved to '{filename}'")

# Save detailed metrics of each fold
save_fold_metrics(metrics_bart)

# ====================== End of Code ======================
