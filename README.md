# CS826
**Course Project for CS826**

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
     $ module load container_env pytorch-gpu/2.2.0
     $ crun -c -p ~/envs/cs826
     ```

4. **Clone the Repository**  
   - Clone this repository to your Wahab home directory:

     ```bash
     $ git clone https://github.com/tstilwell91/cs826.git
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

4. **Download Data**  
   - Navigate to your **Cart** and download the required files:  
     - Click **Download Associated Data -> Metadata**.  
     - Click **Download Cart -> Manifest**.

---

## Step 2: Metadata Preparation

Prepare the metadata for analysis by mapping filenames to their corresponding `case_id`. This mapping creates a common key that is essential for combining the WSI and RNA-seq datasets.

1. Open the `case_mappings.ipynb` notebook in Jupyter.
2. Run the notebook to generate the mapping file.

**Output**: `file_case_mapping.csv`  

This file will be used in subsequent steps of the project.

---

