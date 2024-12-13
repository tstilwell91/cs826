import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Step 1: Define the directory containing the datasets
datasets_dir = '/home/tstil004/phd/multi-omics/rna-seq/'

# Step 2: Load the case mapping file
case_mapping_file = '/home/tstil004/phd/multi-omics/file_case_mapping.csv'
case_mapping_df = pd.read_csv(case_mapping_file)

# Function to process a single RNA-Seq file
def process_rna_seq_file(rna_seq_file, case_mapping_df):
    file = os.path.basename(rna_seq_file)
    print(f"Processing file: {rna_seq_file}")

    # Extract case_id from the filename using the mapping file
    matching_row = case_mapping_df.loc[case_mapping_df['file_name'] == file]
    if matching_row.empty:
        print(f"Skipping file {rna_seq_file} as no matching case_id found in mapping file.")
        return None
    case_id = matching_row['case_id'].values[0]

    # Load the RNA-Seq data into a DataFrame, skipping the comment line
    rna_seq_df = pd.read_csv(rna_seq_file, sep='\t', comment='#')

    # Clean column names to remove leading/trailing whitespace
    rna_seq_df.columns = rna_seq_df.columns.str.strip()

    # Remove rows where 'gene_id' starts with 'N_'
    rna_seq_df = rna_seq_df[~rna_seq_df['gene_id'].str.startswith('N_')]

    # Extract the relevant features ('gene_name' and 'tpm_unstranded')
    if 'gene_name' in rna_seq_df.columns and 'tpm_unstranded' in rna_seq_df.columns:
        rna_features_df = rna_seq_df[['gene_name', 'tpm_unstranded']]
        rna_features_df.insert(0, 'case_id', case_id)  # Add case_id as the first column
    else:
        print(f"Skipping file {rna_seq_file} as it does not contain required columns 'gene_name' and 'tpm_unstranded'")
        return None

    return rna_features_df

# Step 3: Recursively search the directory for TSV files
rna_seq_files = []
for root, _, files in os.walk(datasets_dir):
    for file in files:
        if file.endswith('.tsv'):
            rna_seq_files.append(os.path.join(root, file))

# Step 4: Process files in parallel
all_features_df = pd.DataFrame()
with ProcessPoolExecutor() as executor:
    results = list(executor.map(process_rna_seq_file, rna_seq_files, [case_mapping_df] * len(rna_seq_files)))

# Step 5: Combine all results into a single DataFrame
for result in results:
    if result is not None:
        all_features_df = pd.concat([all_features_df, result], ignore_index=True)

# Step 6: Pivot the DataFrame so that each row is a case_id and each gene_name is a column
all_features_pivot = all_features_df.pivot_table(index='case_id', columns='gene_name', values='tpm_unstranded', aggfunc='mean')
all_features_pivot.reset_index(inplace=True)

# Step 7: Save the combined features to a CSV file
all_features_pivot.to_csv('combined_rna_features2.csv', index=False)

print("RNA-Seq feature extraction complete. Combined features saved to 'combined_rna_features.csv'")

