# CS826
**Course Project for CS826**

---

## Table of Contents
1. [Motivation](#motivation)
2. [Project Overview](#project-overview)
3. [Prerequisites](#prerequisites)
4. [Step 1: Data Retrieval](#step-1-data-retrieval)
5. [Step 2: Metadata Preparation](#step-2-metadata-preparation)
6. [Troubleshooting](#troubleshooting)

---

## Motivation

This project is inspired by recent advancements in computational pathology and genomics integration, particularly in cancer research. The publication "[Integrative analysis of whole slide imaging and RNA-Seq data in prostate cancer](https://www.nature.com/articles/s41598-023-46392-6)" demonstrates the potential for combining RNA-Seq and whole slide imaging (WSI) data to uncover novel insights into prostate cancer. Our work builds upon these ideas to enable efficient data processing and analysis in a high-performance computing environment.

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
   - Transfer these files to your Wahab home directory.

---

## Step 2: Metadata Preparation

Prepare the metadata for analysis by mapping filenames to their corresponding `case_id`. This mapping creates a common key that is essential for combining the WSI and RNA-seq datasets.

1. Open the `case_mappings.ipynb` notebook in Jupyter.
2. Update the `metadata_path` to point to the Metadata file uploaded in Step 1.
3. Specify the output file location.
4. Run the notebook to generate the mapping file.

**Output**: `file_case_mapping.csv`  

This file will be used in subsequent steps of the project.

---

## Troubleshooting

1. **Jupyter Notebook Won’t Launch**  
   - Ensure the cluster partition is set to `gpu`.  
   - Verify that the Python environment is correctly activated with `crun`.

2. **Metadata File Not Found**  
   - Check that the correct path is specified in `case_mappings.ipynb`.

---

