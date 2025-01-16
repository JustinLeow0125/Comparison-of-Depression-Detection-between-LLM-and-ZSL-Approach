import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import pickle

# ==================== Seed Setting for Reproducibility ====================
SEED = 42
np.random.seed(SEED)
# ====================== End of Seed Setting Section ========================

# ==================== Configuration Parameters ====================
DATA_PATH = r'C:\Dataset\Depression Analysis Dataset (DAD).csv'  # Update this path
TEST_SIZE = 0.2
K_FOLDS = 5
FOLD_INDICES_FILE = 'fold_indices.pkl'  # Filename to save fold indices
# ====================== End of Configuration ============================

# ==================== Step 1: Load and Prepare the Dataset ====================
def load_and_prepare_data(data_path):
    df = pd.read_csv(data_path)
    
    required_columns = ['message to examine', 'label']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Dataset must contain '{col}' column.")
    
    # Drop rows with missing values in required columns
    df = df.dropna(subset=required_columns)
    
    # Ensure text column is of type string and not empty
    df['message to examine'] = df['message to examine'].astype(str)
    df = df[df['message to examine'].str.strip() != '']
    
    # Convert labels to integers if they are objects
    if df['label'].dtype == 'object':
        label_mapping = {'Not Depressed': 0, 'Depressed': 1}
        df['label'] = df['label'].map(label_mapping)
    
    # Ensure labels are integers
    df['label'] = df['label'].astype(int)
    
    return df

# Load the dataset
df = load_and_prepare_data(DATA_PATH)
print(f"Total samples after preprocessing: {len(df)}")

# ==================== Step 2: Split into Training+Validation and Test Sets ====================
def split_data(df, test_size=0.2, seed=SEED):
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

train_val_texts, test_texts, train_val_labels, test_labels = split_data(df, TEST_SIZE, SEED)

# ==================== Step 3: Create Fold Indices ====================
def create_fold_indices(texts, labels, k_folds=5, seed=SEED):
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    fold_indices = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        fold_indices.append((train_idx, val_idx))
        print(f"Fold {fold + 1}: Train indices={len(train_idx)}, Validation indices={len(val_idx)}")
    return fold_indices

fold_indices = create_fold_indices(train_val_texts, train_val_labels, K_FOLDS, SEED)

# ==================== Step 4: Save Fold Indices Using Pickle ====================
def save_fold_indices(fold_indices, filename='fold_indices.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(fold_indices, f)
    print(f"Fold indices saved to '{filename}'")

save_fold_indices(fold_indices, FOLD_INDICES_FILE)
