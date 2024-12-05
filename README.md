# CS826
**Course Project for CS826**

---

## Step 1: Data Retrieval

Data for this project is available on the [GDC Data Portal](https://portal.gdc.cancer.gov/projects/TCGA-PRAD). Follow the steps below to retrieve the necessary data:

1. **Access the Project Page**:  
   Navigate to the TCGA-PRAD project page using the link above.

2. **Save a New Cohort**:  
   - Click on **Save New Cohort**.  
   - Provide a name for the cohort and click **Save**.  
   - Click on **Repository** to continue.

3. **Select Data**:  
   - Use the filters on the left-hand menu to select the following:

   **RNA-Seq**:  
   - Under **Experimental Strategy**, select `RNA-Seq`.  
   - Under **Data Category**, select `Transcriptome Profiling`.  
   - Under **Data Type**, select `Gene Expression Quantification`.  
   - Under **Tissue Type**, select `Tumor`.  
   - Click **Add All Files to Cart**.

   **WSI**:  
   - Reset your filters.  
   - Under **Experimental Strategy**, select `Tissue Slide`.  
   - Under **Data Type**, select `Slide Image`.  
   - Under **Tissue Type**, select `Tumor`.  
   - Click **Add All Files to Cart**.

4. **Download Data**:  
   - Click on **Cart** at the top of the screen.  
   - Download the required files:
     - Click **Download Associated Data -> Metadata**.  
     - Click **Download Cart -> Manifest**.

---

## Step 2: Metadata Preparation

To prepare the data for analysis, we need to map each filename in the TCGA-PRAD dataset to its corresponding `case_id`. This mapping allows us to create a common key, which will be used to combine the WSI and RNA-seq datasets effectively.

---

