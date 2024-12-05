# CS826
**Course Project for CS826**

---

## Table of Contents
1. [Motivation](#motivation)
2. [Project Overview](#project-overview)
3. [Prerequisites](#prerequisites)
4. [Step 1: Data Retrieval](#step-1-data-retrieval)
5. [Step 2: Metadata Preparation](#step-2-metadata-preparation)
6. [Step 3: RNA-Seq Feature Extraction](#step-3-rna-seq-feature-extraction)
7. [Step 4: WSI Feature Extraction](#step-4-wsi-feature-extraction)
8. [Step 5: Concatenate Data](#step-5-concatenate-data)
9. [Improvements](#improvements)

---

## Motivation

This project is inspired by recent advancements in computational pathology and genomics integration, particularly in cancer research. The publication "[Integrative analysis of whole slide imaging and RNA-Seq data in prostate cancer](https://www.nature.com/articles/s41598-023-46392-6)" demonstrates the potential for combining RNA-Seq and whole slide imaging (WSI) data to uncover novel insights into prostate cancer. Our work will build upon these ideas to include genomic data and telomere features for prostate cancer.

---

## Project Overview

This project involves analyzing RNA-Seq and Whole Slide Imaging (WSI) data from the TCGA-PRAD dataset. The goal is to preprocess and combine these datasets for further analysis, leveraging computational resources on the ODU Wahab cluster.

---

## Prerequisites

All work for this project is performed on the ODU Wahab cluster. Follow the steps below to set up your environment:

1. **Cluster Access**  
   If you do not have access to the Wahab cluster, fill out the access request form:  
   [ODU Wahab Access Form](https://forms.odu.edu/view.php?id=93440).

2. **Login to the Cluster**  
   - For interactive work, access the Wahab OnDemand portal:  
     [ODU Wahab OnDemand](https://ondemand.wahab.hpc.odu.edu).

3. **Set Up Python Prerequisites**  
   - Connect to the cluster over SSH or via the OnDemand portal under **Cluster Access**.  
   - Load the necessary modules and activate the Python environment:

     ```bash
     module load container_env pytorch-gpu/2.2.0
     crun -c -p ~/envs/cs826
     ```

4. **Clone the Repository**  
   - Clone this repository to your Wahab home directory:

     ```bash
     git clone https://github.com/tstilwell91/cs826.git
     ```

5. **Launch Jupyter on Wahab**  
   - From the OnDemand portal, navigate to **Interactive Apps -> Jupyter** and configure the following options:
     - **Python Version**: Python 3.10
     - **Python Suite**: PyTorch 2.2.0
     - **Additional Module Directory**: `/home/tstil004/envs/cs826`
     - **Number of Cores**: 8
     - **Number of GPUs**: 1
     - **Partition**: gpu
   - Click **Launch**.  
   - Once the job is ready, click **Connect to Jupyter**.

---

## Step 1: Data Retrieval

Retrieve the necessary data for this project from the [GDC Data Portal](https://portal.gdc.cancer.gov/projects/TCGA-PRAD):

### Data Information
- **Total Files**: 1,107
- **Total Size**: 133.4 GB
- **Number of Cases**: 498

### Instructions
1. **Access the Project Page**  
   Navigate to the TCGA-PRAD project page using the link above.

2. **Save a New Cohort**  
   - Click **Save New Cohort**, provide a name, and click **Save**.  
   - Then, click **Repository** to continue.

3. **Select Data**  

   **RNA-Seq**:  
   - Under **Experimental Strategy**, select `RNA-Seq`.  
   - Under **Data Category**, select `Transcriptome Profiling`.  
   - Under **Data Type**, select `Gene Expression Quantification`.  
   - Under **Tissue Type**, select `Tumor`.  
   - Click **Add All Files to Cart**.

   **WSI (Whole Slide Imaging)**:  
   - Reset your filters.  
   - Under **Experimental Strategy**, select `Tissue Slide`.  
   - Under **Data Type**, select `Slide Image`.  
   - Under **Tissue Type**, select `Tumor`.  
   - Click **Add All Files to Cart**.

4. **Download and Upload Data**
   - Navigate to your **Cart** and download the required files:
     - **Metadata**: Click **Download Associated Data -> Metadata**.
     - **Manifest**: Click **Download Cart -> Manifest**.
   - Download your **User Token**: Click on your Profile and select **Download Token**.
   - Transfer the downloaded files (Metadata, Manifest, and Token) to your Wahab home directory.

5. **Download RNA-Seq and WSI Files Using gdc-client**
   - Use the `gdc-client` to download the files listed in the manifest. Replace the file paths in the command below with the appropriate paths for your manifest and token.  You can download all the data at once or create two separate manifests to download the data separately.

   ### Command Example

     ```bash
     $ gdc-client download -m gdc_manifest.txt -t gdc-user-token.txt
     ```

   - **Note**: The `gdc-client` is not installed centrally. Its location on the Wahab cluster is:
     `/home/tstil004/gdc-client/gdc-client`

   **Example Directory Structure**:
   ```plaintext
   /path/to/datasets/
   ├── rna-seq/
   │   ├── directory1/
   │   │   └── rna-seq1.tsv
   │   ├── directory2/
   │       └── rna-seq2.tsv
   ├── wsi/
   │   ├── directory1/
   │   │   └── wsi1.svs
   │   ├── directory2/
   │       └── wsi2.svs
   ```
---

## Step 2: Metadata Preparation

Prepare the metadata for analysis. 

1. Open the `case_mappings.ipynb` notebook in Jupyter.
2. Update `metadata_path` to point to the Metadata file uploaded in Step 1.
3. Specify the output file location.
4. Run the notebook to generate the mapping file.

**Output**: `file_case_mapping.csv`  

This file contains the mapping of filenames to `case_id` and serves as a crucial key for combining the RNA-Seq and WSI datasets. This will be used as input in future steps of the project.

---

## Step 3: RNA-Seq Feature Extraction

A separate manifest was created for the RNA-Seq data, allowing it to be downloaded independently from the WSI data.

### Instructions
1. Open the `rna-seq-features-extraction.ipynb` notebook in Jupyter.
2. Update the `datasets_dir` variable to point to the directory containing the RNA-Seq files downloaded earlier using the `gdc-client`.
3. Update the `case_mapping_file` variable to reference the file generated in **Step 2**.
4. Run the notebook to process the RNA-Seq data and generate the RNA-Seq features file.

**Output**: `combined_rna_features.csv`  

This file contains the extracted gene expression features mapped to their corresponding `case_id`.

---

## Step 4: WSI Feature Extraction

A separate manifest was created for the WSI data, allowing it to be downloaded independently from the RNA-Seq data.

### Instructions

1. Open the `wsi-features-extraction.ipynb` notebook in Jupyter.
2. Update the `DATA_DIR` variable to point to the directory containing the WSI files downloaded using the `gdc-client`.
3. Update the `CASE_MAPPING_FILE` variable to reference the file generated in **Step 2**.
4. Set the following parameters as used in the referenced paper:
   - `EPOCHS = 100`
   - `NUM_TILES = 128`
   - `FEATURE_DIM = 512`
5. Adjust the `BATCH_SIZE` and `NUM_WORKERS` values based on the available computational resources.
6. Run the notebook to process the WSI data and generate the WSI features file.

**Output**: `extracted_wsi_features.csv`

This file contains the extracted WSI features mapped to their corresponding `case_id`.

---

## Step 5: Concatenate Data

Combine the RNA-Seq and WSI datasets into a single features file.

1. Open the `concat.ipynb` notebook in Jupyter.
2. Update the paths for `rna_features` and `wsi_features` to point to the output files generated in the previous steps.
3. Run the notebook to generate the combined features file.

**Output**: `combined_features.csv`  

This file contains the combined features for each `case_id` and will be used in the next step to generate the Individual Networks (IN).

---

## Improvements  
Rather than running these scripts individually to generate separate CSV files, it would be more efficient to integrate them into a single step using data frames. This approach not only simplifies the workflow but also enhances performance.
