import json
import pandas as pd

# Load metadata JSON file
metadata_path = "/home/tstil004/phd/multi-omics/metadata.cart.2024-11-11.json"
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# Extract case_id and file_name
mapping = []
for entry in metadata:
    case_id = entry.get("associated_entities")[0].get("case_id")
    file_name = entry.get("file_name")
    mapping.append({"file_name": file_name, "case_id": case_id})

# Create a DataFrame for easier handling
mapping_df = pd.DataFrame(mapping)

# Save mapping to CSV for later use
mapping_df.to_csv("/home/tstil004/phd/multi-omics/file_case_mapping2.csv", index=False)
