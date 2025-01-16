# ==================== Import Necessary Libraries ====================
import pandas as pd  # For data manipulation and analysis
import nltk  # For natural language processing tasks
import nlpaug.augmenter.word as naw  # For data augmentation of text at the word level
from transformers import (
    BertTokenizerFast,  # Fast BERT tokenizer
    BertForSequenceClassification,  # BERT model for sequence classification
    Trainer,  # Hugging Face Trainer class for training/evaluation loops
    TrainingArguments,  # Configuration for training
    TrainerCallback,  # Base class for custom callbacks
    BertConfig  # Configuration class for BERT model
)
from sklearn.model_selection import train_test_split  # To split dataset into training/validation/test sets
from sklearn.metrics import (
    accuracy_score,  # For calculating accuracy
    precision_recall_fscore_support,  # For precision, recall, F1 metrics
    roc_auc_score,  # For AUC-ROC metric
    classification_report,  # For a detailed classification metrics report
    confusion_matrix,  # For generating confusion matrix
    roc_curve,  # For ROC curve calculation
    auc  # To compute area under ROC curve
)
from sklearn.utils.class_weight import compute_class_weight  # To compute class weights for imbalanced data
import torch  # PyTorch framework for deep learning
import torch.nn as nn  # Neural network layers, loss functions, etc.
import numpy as np  # Numerical operations
import warnings  # To manage and suppress warnings
import matplotlib.pyplot as plt  # For plotting charts and graphs
import seaborn as sns  # For advanced plotting (heatmaps, etc.)
import random  # For setting random seeds and random operations
import pickle  # For loading/saving Python objects (here used for fold_indices)
import os  # For path handling and file existence checks

# Suppress any warnings to keep output clean
warnings.filterwarnings('ignore')

# ==================== Seed Setting for Reproducibility ====================
def set_seed(seed: int = 42):
    # Sets various random seeds to ensure reproducible results
    random.seed(seed)  # Python's built-in random seed
    np.random.seed(seed)  # NumPy's random seed
    torch.manual_seed(seed)  # PyTorch CPU random seed
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU random seed (for all GPUs)
    torch.backends.cudnn.deterministic = True  # Ensures determinism for convolutional ops
    torch.backends.cudnn.benchmark = False  # Disables benchmarking for determinism (may slow down training)

SEED = 42
set_seed(SEED)
# ====================== End of Seed Setting Section ========================

# Determine if GPU is available and set the device accordingly
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Step 1: Load and prepare the dataset
df = pd.read_csv(r'C:\Dataset\Depression Analysis Dataset (DAD).csv')  # Load dataset into a DataFrame

required_columns = ['message to examine', 'label']  # Columns that must be present
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Dataset must contain '{col}' column.")

# Drop rows with missing required fields
df = df.dropna(subset=required_columns)
df['message to examine'] = df['message to examine'].astype(str)  # Ensure text column is string
df = df[df['message to examine'].str.strip() != '']  # Remove empty messages

# Convert labels from strings to integers if needed
if df['label'].dtype == 'object':
    label_mapping = {'Not Depressed': 0, 'Depressed': 1}
    df['label'] = df['label'].map(label_mapping)

# Ensure labels are integer type
if df['label'].dtype not in ['int64', 'int32', 'int']:
    df['label'] = df['label'].astype(int)

# Step 2: Split the dataset into Training+Validation and Test sets
TEST_SIZE = 0.2  # Percentage of data for test set
train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
    df['message to examine'].tolist(),
    df['label'].tolist(),
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=df['label']  # Ensure balanced label distribution
)

print(f"Training + Validation samples: {len(train_val_texts)}")
print(f"Test samples: {len(test_texts)}")

# Step 3: Load Fold Indices from 'fold_indices.pkl'
FOLD_INDICES_PATH = r'C:\Dataset\fold_indices.pkl'  # Path to fold indices file

if not os.path.exists(FOLD_INDICES_PATH):
    # If fold_indices.pkl doesn't exist, generate new fold indices
    from sklearn.model_selection import StratifiedKFold
    k_folds = 5  # Number of folds for cross-validation
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    fold_indices = list(skf.split(train_val_texts, train_val_labels))

    # Save the generated fold indices
    with open(FOLD_INDICES_PATH, 'wb') as f:
        pickle.dump(fold_indices, f)

    print(f"Stratified K-Fold indices generated and saved to {FOLD_INDICES_PATH}")
else:
    # Load existing fold indices
    with open(FOLD_INDICES_PATH, 'rb') as f:
        fold_indices = pickle.load(f)
    k_folds = len(fold_indices)
    print(f"Loaded fold indices from {FOLD_INDICES_PATH}")

texts = train_val_texts
labels = train_val_labels

# Step 4: Initialize BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Step 5: Define a custom Dataset class for the model
class DepressionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings  # Tokenized inputs
        self.labels = labels  # Corresponding labels

    def __getitem__(self, idx):
        # Retrieve a single sample
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        # Return total number of samples
        return len(self.labels)

# Step 6: Define a function to compute evaluation metrics
def compute_metrics(p):
    preds = p.predictions  # Model predictions (logits)
    labels = p.label_ids   # True labels
    # Convert logits to probabilities using softmax
    probs = torch.nn.functional.softmax(torch.tensor(preds), dim=-1)
    # Predicted classes are the argmax of logits
    preds_class = np.argmax(preds, axis=1)

    # Compute precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds_class, average='binary')
    acc = accuracy_score(labels, preds_class)

    # Compute AUC-ROC (may fail if there's only one class in predictions)
    try:
        auc_roc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auc_roc = float('nan')

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': auc_roc,
    }

# Step 7: Data collator to create batches for the Trainer
def data_collator(features):
    # Stack all tensor elements from the list of features into a batch
    batch = {}
    batch['input_ids'] = torch.stack([f['input_ids'] for f in features])
    batch['attention_mask'] = torch.stack([f['attention_mask'] for f in features])
    batch['labels'] = torch.stack([f['labels'] for f in features])

    # If token_type_ids are present (for BERT), include them as well
    if 'token_type_ids' in features[0]:
        batch['token_type_ids'] = torch.stack([f['token_type_ids'] for f in features])

    return batch

# Step 8: Early Stopping Callback to halt training if no improvement
class CustomEarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience=1):
        self.early_stopping_patience = early_stopping_patience
        self.best_metric = None
        self.patience_counter = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Called after each evaluation
        if metrics is None:
            return
        current_metric = metrics.get('eval_accuracy')
        if self.best_metric is None or current_metric > self.best_metric:
            # Improvement found, reset patience counter
            self.best_metric = current_metric
            self.patience_counter = 0
        else:
            # No improvement, increment counter
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                # Stop training
                print("Early stopping triggered")
                control.should_terminate_training = True

early_stopping_callback = CustomEarlyStoppingCallback(early_stopping_patience=1)

# Step 9: Initialize the Data Augmenter for text augmentation
nltk.download('wordnet')  # For SynonymAug augmentation
nltk.download('omw-1.4')  # Additional wordnet data
nltk.download('averaged_perceptron_tagger')  # For POS tagging in synonym augmentation

synonym_aug = naw.SynonymAug(aug_p=0.1)  # Augment 10% of words with synonyms

def augment_texts(texts, augmenter, num_augmented_versions=1):
    # Apply text augmentation to each text
    augmented_texts = []
    for text in texts:
        augmented = augmenter.augment(text, n=num_augmented_versions)
        # If augmenter returns a list, extend augmented_texts; else append single item
        if isinstance(augmented, list):
            augmented_texts.extend(augmented)
        else:
            augmented_texts.append(augmented)
    return augmented_texts

# Step 10: Custom Trainer to incorporate class weights into the loss
class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Overriding loss computation to include class weights
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
        else:
            return outputs

# Step 11: Initialize a list to store fold-wise metrics
fold_metrics = []

# Step 12: Cross-Validation Loop using the precomputed fold indices
for fold, (train_index, val_index) in enumerate(fold_indices):
    print(f"\n========== Fold {fold + 1}/{k_folds} ==========")
    
    # Separate training and validation samples for this fold
    train_texts_fold = [texts[i] for i in train_index]
    train_labels_fold = [labels[i] for i in train_index]
    val_texts_fold = [texts[i] for i in val_index]
    val_labels_fold = [labels[i] for i in val_index]

    # Augment training texts to increase data diversity
    augmented_train_texts = augment_texts(train_texts_fold, synonym_aug, num_augmented_versions=1)

    # Combine original and augmented data
    combined_train_texts = train_texts_fold + augmented_train_texts
    combined_train_labels = train_labels_fold + train_labels_fold  # Duplicate labels for augmented texts

    print(f"Fold {fold + 1} - Original training samples: {len(train_texts_fold)}")
    print(f"Fold {fold + 1} - Augmented training samples: {len(augmented_train_texts)}")
    print(f"Fold {fold + 1} - Combined training samples: {len(combined_train_texts)}")
    print(f"Fold {fold + 1} - Combined training labels: {len(combined_train_labels)}")

    # Tokenize combined training data
    combined_train_encodings = tokenizer(combined_train_texts, truncation=True, padding=True, max_length=128)
    combined_train_dataset = DepressionDataset(combined_train_encodings, combined_train_labels)

    # Tokenize validation data
    val_encodings_fold = tokenizer(val_texts_fold, truncation=True, padding=True, max_length=128)
    val_dataset_fold = DepressionDataset(val_encodings_fold, val_labels_fold)

    # Compute class weights to handle imbalanced data
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(combined_train_labels),
        y=combined_train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Configure and initialize the BERT model
    config = BertConfig.from_pretrained('bert-base-uncased')
    config.hidden_dropout_prob = 0.3
    config.attention_probs_dropout_prob = 0.3
    config.num_labels = 2

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        config=config
    ).to(device)

    print("Some weights of BertForSequenceClassification were not initialized... (Normal for newly initialized classifier layers)")

    # Freeze first N layers of BERT to speed up and stabilize training
    N = 6
    for layer in model.bert.encoder.layer[:N]:
        for param in layer.parameters():
            param.requires_grad = False

    print(f"Frozen the first {N} layers of BERT.")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f'./results/fold_{fold + 1}',  # Directory to store model outputs
        num_train_epochs=3,  # Number of training epochs
        per_device_train_batch_size=16,  # Batch size for training
        per_device_eval_batch_size=64,   # Batch size for evaluation
        eval_strategy='epoch',  # Evaluate after each epoch
        save_strategy='epoch',  # Save model after each epoch
        save_total_limit=1,     # Keep only the most recent checkpoint
        logging_dir=f'./logs/fold_{fold + 1}',  # Logging directory
        logging_steps=10,  # Log every 10 steps
        load_best_model_at_end=True,  # Load the best model at the end of training
        metric_for_best_model='accuracy',  # Use accuracy to determine best model
        greater_is_better=True,  # Higher metric is better
        learning_rate=2e-5,  # Initial learning rate
        weight_decay=0.01,   # Weight decay for regularization
        warmup_steps=100,    # Number of warmup steps for the scheduler
        lr_scheduler_type='linear',  # Learning rate scheduler type
        evaluation_strategy='epoch', # Evaluate every epoch (redundant since eval_strategy already set)
        seed=SEED,  # Random seed for reproducibility
    )

    # Initialize the custom trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=combined_train_dataset,
        eval_dataset=val_dataset_fold,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        class_weights=class_weights,
        callbacks=[early_stopping_callback]
    )

    # Train the model
    trainer.train()

    # Evaluate the model on validation set
    eval_results = trainer.evaluate()
    print("\nValidation Set Evaluation Results:")
    for key, value in eval_results.items():
        if key.startswith('eval_'):
            print(f"{key}: {value:.4f}")

    def generate_classification_report_fold(dataset, labels, set_name="Validation"):
        # Generate predictions and a classification report for the given dataset
        predictions_output = trainer.predict(dataset)
        preds_class = np.argmax(predictions_output.predictions, axis=1)
        report = classification_report(labels, preds_class, target_names=['Not Depressed', 'Depressed'])
        print(f"\nClassification Report for {set_name} Set (Fold {fold + 1}):")
        print(report)
        return report

    # Classification report for training set (with augmented data)
    generate_classification_report_fold(combined_train_dataset, combined_train_labels, set_name="Training")
    # Classification report for validation set
    generate_classification_report_fold(val_dataset_fold, val_labels_fold, set_name="Validation")

    # Save the trained model and tokenizer for this fold
    fold_model_save_path = f'./depression_classifier_model_fold_{fold + 1}'
    trainer.model.save_pretrained(fold_model_save_path)
    tokenizer.save_pretrained(fold_model_save_path)
    print(f"\nModel and tokenizer saved to {fold_model_save_path}")

    # Store metrics for this fold
    fold_metrics.append({
        'fold': fold + 1,
        'eval_loss': eval_results.get('eval_loss'),
        'eval_accuracy': eval_results.get('eval_accuracy'),
        'eval_precision': eval_results.get('eval_precision'),
        'eval_recall': eval_results.get('eval_recall'),
        'eval_f1': eval_results.get('eval_f1'),
        'eval_roc_auc': eval_results.get('eval_roc_auc'),
    })

# Convert fold metrics to a DataFrame and print results
metrics_df = pd.DataFrame(fold_metrics)
print("\n========== Cross-Validation Results ==========")
print(metrics_df)

# Compute average and standard deviation of metrics across folds
avg_metrics = metrics_df.mean()
std_metrics = metrics_df.std()

print("\nAverage Metrics Across All Folds:")
for metric in ['eval_loss', 'eval_accuracy', 'eval_precision', 'eval_recall', 'eval_f1', 'eval_roc_auc']:
    print(f"{metric}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")

# Plot metrics across folds
plt.figure(figsize=(10, 6))
for metric in ['eval_accuracy', 'eval_precision', 'eval_recall', 'eval_f1', 'eval_roc_auc']:
    plt.plot(metrics_df['fold'], metrics_df[metric], marker='o', label=metric.replace('eval_', '').capitalize())

plt.xlabel('Fold')
plt.ylabel('Score')
plt.title('Validation Metrics Across Folds')
plt.legend()
plt.grid(True)
plt.show()

print("\n========== Final Model Training on Entire Training+Validation Dataset ==========")
# Train final model on all training+validation data
final_train_texts = texts
final_train_labels = labels

# Augment the final training data
augmented_final_train_texts = augment_texts(final_train_texts, synonym_aug, num_augmented_versions=1)
combined_final_train_texts = final_train_texts + augmented_final_train_texts
combined_final_train_labels = final_train_labels + final_train_labels

print(f"Final Training - Original samples: {len(final_train_texts)}")
print(f"Final Training - Augmented samples: {len(augmented_final_train_texts)}")
print(f"Final Training - Combined samples: {len(combined_final_train_texts)}")
print(f"Final Training - Combined labels: {len(combined_final_train_labels)}")

# Tokenize final training data
combined_final_train_encodings = tokenizer(combined_final_train_texts, truncation=True, padding=True, max_length=128)
combined_final_train_dataset = DepressionDataset(combined_final_train_encodings, combined_final_train_labels)

# Compute class weights for final training
final_class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(combined_final_train_labels),
    y=combined_final_train_labels
)
final_class_weights = torch.tensor(final_class_weights, dtype=torch.float).to(device)

# Configure and initialize the final model
final_config = BertConfig.from_pretrained('bert-base-uncased')
final_config.hidden_dropout_prob = 0.3
final_config.attention_probs_dropout_prob = 0.3
final_config.num_labels = 2

final_model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    config=final_config
).to(device)

print("Some weights... newly initialized. You should train this model downstream...")

# Freeze first N layers for the final model as well
N = 6
for layer in final_model.bert.encoder.layer[:N]:
    for param in layer.parameters():
        param.requires_grad = False

print(f"Frozen the first {N} layers of BERT for the final model.")

final_training_args = TrainingArguments(
    output_dir='./results/final_model',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    eval_strategy='no',  # No separate evaluation during training
    save_strategy='epoch',
    save_total_limit=1,
    logging_dir='./logs/final_model',
    logging_steps=10,
    load_best_model_at_end=False,
    metric_for_best_model='accuracy',
    greater_is_better=True,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,
    lr_scheduler_type='linear',
    evaluation_strategy='no',  # No evaluation during training
    seed=SEED,
)

final_trainer = CustomTrainer(
    model=final_model,
    args=final_training_args,
    train_dataset=combined_final_train_dataset,
    eval_dataset=None,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    class_weights=final_class_weights,
    callbacks=[]
)

# Train final model on the combined training+validation data
final_trainer.train()

# Prepare test dataset
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)
test_dataset = DepressionDataset(test_encodings, test_labels)

# Evaluate final model on test data
final_test_results = final_trainer.evaluate(test_dataset)
print("\nFinal Test Set Evaluation Results:")
for key, value in final_test_results.items():
    if key.startswith('eval_'):
        print(f"{key}: {value:.4f}")

def generate_classification_report_and_confusion_matrix(dataset, labels, set_name="Test"):
    # Generate predictions on the specified dataset
    predictions_output = final_trainer.predict(dataset)
    preds_class = np.argmax(predictions_output.predictions, axis=1)
    # Compute probabilities for ROC curve
    probs = torch.nn.functional.softmax(torch.tensor(predictions_output.predictions), dim=-1).numpy()

    # Print classification report
    report = classification_report(labels, preds_class, target_names=['Not Depressed', 'Depressed'])
    print(f"\nClassification Report for {set_name} Set:")
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(labels, preds_class)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Depressed', 'Depressed'],
                yticklabels=['Not Depressed', 'Depressed'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix for {set_name} Set')
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(labels, probs[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for {set_name} Set')
    plt.legend(loc="lower right")
    plt.show()

generate_classification_report_and_confusion_matrix(test_dataset, test_labels, set_name="Test")

# Save final model and tokenizer
final_model_save_path = './depression_classifier_final_model'
final_trainer.model.save_pretrained(final_model_save_path)
tokenizer.save_pretrained(final_model_save_path)
print(f"\nFinal model and tokenizer saved to {final_model_save_path}")

def predict_final(texts):
    # Predict depression vs not depression on new texts
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
    encodings = {key: val.to(device) for key, val in encodings.items()}
    final_trainer.model.eval()
    with torch.no_grad():
        outputs = final_trainer.model(**encodings)
        logits = outputs['logits']
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
    return predictions.cpu().numpy(), probs.cpu().numpy()

new_texts = [
    "I'm depressed",
    "What a wonderful day!",
    "I can't handle this stress anymore.",
    "Excited for the holidays."
]

predictions, probabilities = predict_final(new_texts)
label_map = {0: 'Not Depressed', 1: 'Depressed'}
for text, pred, prob in zip(new_texts, predictions, probabilities):
    print(f"Text: {text}\nPrediction: {label_map[pred]} (Confidence: {prob[pred]:.4f})\n")
