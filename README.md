## Current Progress

### Data Acquisition and Preprocessing

- Google Earth Engine configured and project registered
- Hyderabad development region defined
- Sentinel-2 SR Harmonized data selected
- Sentinel-2 imagery collected for 2020 and 2025
- Cloud and cirrus masking implemented
- Median composites generated for both temporal periods
- RGB composites visually validated
- Band consistency and valid-pixel checks completed

### Spectral Feature Engineering

- NDVI generation implemented for 2020 and 2025
- NDWI generation implemented for 2020 and 2025
- NDBI generation implemented for 2020 and 2025
- Temporal difference features generated (ΔNDVI, ΔNDWI, ΔNDBI)
- Bi-temporal spectral feature stack created
- Spectral feature maps and statistics validated

### LULC and Dataset Preparation

- Reference LULC data incorporated for baseline development
- Reference classes mapped to project-level LULC categories
- Four project-level LULC classes defined:
  - Water
  - Vegetation
  - Built-up
  - Bare Land
- Corrected 2020 reference labels generated
- 2025 reference labels generated
- 2020 and 2025 labels aligned with Sentinel-2 image grids
- 128×128 bi-temporal image patches prepared
- 16 semantic transition classes generated
- Binary change labels generated
- Consistent train/validation/test split prepared
- Final v4 dataset validated and published

### Classical Machine-Learning Baselines

- Random Forest baseline implemented
- Support Vector Machine (SVM) baseline implemented
- XGBoost baseline implemented
- Accuracy, Precision, Recall and F1-score evaluated
- SVM achieved the highest overall accuracy among the classical baselines

### Deep-Learning Baselines

- Early-Fusion CNN implemented and trained
- Siamese CNN implemented and trained
- Siamese U-Net implemented and trained
- Common 9-channel bi-temporal inputs used across all models
- 16-class semantic transition output implemented
- 2-class binary change output implemented
- Consistent train/validation/test split used across all models
- Test-set evaluation completed
- 16×16 semantic transition confusion matrices generated
- 2×2 binary change confusion matrices generated
- Representative prediction visualizations generated
- Baseline models and evaluation results saved

### Proposed Framework

- Develop the proposed bi-temporal deep-learning architecture
- Integrate spatial and temporal feature learning
- Incorporate NDVI, NDWI and NDBI with learned features
- Implement semantic land-cover transition detection
- Develop explainability/XAI module
- Compare proposed model against classical and deep-learning baselines
- Evaluate using Accuracy, Precision, Recall, F1-score and IoU/mIoU
- Generate semantic change maps
- Perform ablation studies
- Evaluate cross-region generalization
- Analyze model errors and failure cases
