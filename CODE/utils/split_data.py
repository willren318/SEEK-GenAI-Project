import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Create output directory if it doesn't exist
output_dir = "MISC/split_seniority-dev-data_into_train_validatoin"
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
data_path = "MISC/job_data_files/seniority_labelled_development_set.csv"
df = pd.read_csv(data_path)

# Find classes with only 1 example
class_counts = df['y_true'].value_counts()
rare_classes = class_counts[class_counts < 2].index.tolist()

# Split the data
if rare_classes:
    # Put rare class examples in the training set
    rare_samples = df[df['y_true'].isin(rare_classes)]
    
    # Split the rest with stratification
    rest_samples = df[~df['y_true'].isin(rare_classes)]
    train_rest, val_df = train_test_split(rest_samples, test_size=0.2, random_state=42, stratify=rest_samples['y_true'])
    
    # Combine rare samples with training set
    train_df = pd.concat([train_rest, rare_samples])
else:
    # Regular stratified split if no rare classes
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['y_true'])

# Save the split datasets
train_df.to_csv(f"{output_dir}/train_set.csv", index=False)
val_df.to_csv(f"{output_dir}/validation_set.csv", index=False)

print(f"Data split complete. Train set: {len(train_df)} examples. Validation set: {len(val_df)} examples.")
print(f"Files saved to {output_dir}/")
