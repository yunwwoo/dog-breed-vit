# Dog Breed Identifier (ViT Fine-Tuning)

Fine-tunes a pretrained Vision Transformer (google/vit-base-patch16-224) on the Stanford Dogs dataset (120 breeds, ~20K images) for breed classification, with a Streamlit app for interactive predictions.

## Project structure
- `notebooks/01_data_pipeline_and_smoketest.ipynb` — local data loading and pipeline sanity check
- `notebooks/02_train.ipynb` — full fine-tuning run on Google Colab (T4 GPU)
- `app/app.py` — Streamlit web app for uploading a photo and getting breed predictions

## Results
Final validation accuracy: 87.3% across 120 breeds after 4 epochs of fine-tuning.

## Setup
See requirements and run instructions in each notebook / app file.
