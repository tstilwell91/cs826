import pandas as pd

# Load the RNA-Seq features and WSI features
rna_features = pd.read_csv('/home/tstil004/phd/multi-omics/rna-seq/combined_rna_features.csv')
wsi_features = pd.read_csv('/home/tstil004/phd/multi-omics/slides/extracted_wsi_features.csv')

# Merge the two datasets on 'case_id'
combined_features = pd.merge(rna_features, wsi_features, on='case_id', how='inner')

# Save the combined features to a new CSV file
combined_features.to_csv('combined_features.csv', index=False)

print("Combined features saved to 'combined_features.csv'")
