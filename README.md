## Current Progress

### Data Acquisition and Preprocessing

-  Google Earth Engine configured and project registered
-  Hyderabad development region defined
-  Sentinel-2 SR Harmonized data selected
-  Sentinel-2 imagery collected for 2020 and 2025
-  Cloud and cirrus masking implemented
-  Median composites generated for both temporal periods
-  RGB composites visually validated
-  Band consistency and valid-pixel checks completed

### Spectral Feature Engineering

-  NDVI generation implemented for 2020 and 2025
-  NDWI generation implemented for 2020 and 2025
-  NDBI generation implemented for 2020 and 2025
-  Temporal difference features generated (ΔNDVI, ΔNDWI, ΔNDBI)
-  Bi-temporal spectral feature stack created
-  Spectral feature maps and statistics validated

### LULC Baseline Preparation

-  Reference LULC data incorporated for baseline development
-  Reference classes mapped to project-level LULC categories
-  Training samples generated using stratified sampling
-  Training and testing datasets prepared
-  Four LULC classes considered:
  - Water
  - Vegetation
  - Built-up
  - Bare Land

### Proposed Framework

-  Develop the proposed bi-temporal deep-learning architecture
-  Integrate spatial and temporal feature learning
-  Incorporate NDVI, NDWI and NDBI with learned features
-  Implement semantic land-cover transition detection
-  Develop explainability/XAI module


-  Compare proposed model against classical and deep-learning baselines
-  Evaluate using Accuracy, Precision, Recall, F1-score and IoU/mIoU
-  Generate semantic change maps
-  Perform ablation studies
-  Evaluate cross-region generalization
-  Analyze model errors and failure cases
