# BT-SSSCNet Streamlit Demo

Inference/presentation app around the **frozen, already-trained** BT-SSSCNet. No training, no new preprocessing,
no new normalisation statistics.

## Install (Windows PowerShell)
```powershell
cd BT_SSSCNet_Streamlit
.\.venv\Scripts\Activate.ps1          # create first if needed: py -3.11 -m venv .venv
pip install -r requirements.txt
```

## Files
```
app.py  analysis.py  model_utils.py  preprocessing.py  visualization.py  visualization_ext.py  xai_utils.py
verify_against_notebook.py  self_test.py  requirements.txt  README.md
model/bt_ssscnet_best_val_loss.keras
sample_data/  (samples_index.json norm_stats.npz preprocessing_config.json class_mappings.json model_hash.json + sample folders)
              optional: lulc_names.json  {"0": "...", "1": "...", "2": "...", "3": "..."}
```
Class names are not part of the dataset; without a documented `lulc_names.json` the app shows `LULC 0..3`.

## Input format
Two co-registered `.npy` patches (earlier 2020, later 2025), shape (9,128,128) or (128,128,9), real, finite,
channels [B2, B3, B4, B8, B11, B12, NDVI, NDWI, NDBI] in the dataset's RAW value ranges. The app normalises exactly once.
JPG/PNG/GeoTIFF/RGB images and other sizes are rejected: they lack the bands the model and its indices need.

## Run
```powershell
python self_test.py                   # logic tests + model/sample/XAI smoke tests
python verify_against_notebook.py     # hash + frozen-prediction check
streamlit run app.py
```

## Reading the dashboard
- Sidebar metrics = model-wide notebook evaluation (author-reported, not recomputed).
- "Ground truth" tab metrics = one bundled patch only.
- Percentages are pixel shares over the analysed pixels, not area.
- Spectral-index trends are not land-cover area changes. XAI views are descriptive, not causal.

## Troubleshooting
- Hash MISMATCH: re-download the exact model file recorded in `sample_data/model_hash.json`; never edit the JSON to force a match.
- Model will not load: install the TensorFlow version in `preprocessing_config.json` (`tf_version`).
- XAI warning: see "Layer availability" in the Explainability tab.
- Slow Integrated Gradients: use 8-16 steps.
