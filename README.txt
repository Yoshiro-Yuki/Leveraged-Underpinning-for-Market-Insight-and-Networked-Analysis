LUMINA (Leveraged Underpinning for Market Insight and Networked Analysis)

An end-to-end machine learning system designed to optimize customer-to-customer (C2C) e-commerce listings. This application utilizes a stacked ensemble architecture to predict the optimal listing price and a multi-stage gradient boosting classifier to simulate buyer clickstream behavior (Attraction, Interest, Conversion).

Architecture Overview

The system is orchestrated by a central `MetaModel` that chains two distinct machine learning pipelines:

1. PurchaseMotivation (Price Suggestor): A two-level stacked regression model.
   - Level 1: TF-IDF + Ridge Regression (Textual features) & LightGBM Regressor (Categorical/Numerical features with Out-Of-Fold Target Encoding).
   - Level 2: Ridge Regression meta-learner combining Level 1 outputs to generate a final logarithmic price prediction.
2. ClickstreamAnalysis (Behavioral Flow): Three independent LightGBM binary classifiers predicting the probability of user funnel progression:
   - View (Attraction)
   - Like/Add to Cart (Interest)
   - Buy/Offer (Conversion)
   
The UI (Streamlit) utilizes these models to dynamically calculate feature impacts (Brand, Condition, Shipping equity) and plot simulated demand curves by sweeping price variables.

Prerequisites & Dependencies

Ensure you have Python 3.8+ installed. The following libraries are required:
- `pandas`
- `numpy`
- `scikit-learn`
- `lightgbm`
- `joblib`
- `streamlit`
- `plotly`
- `scipy`
- `imblearn`

Getting Started

1. Data Requirements
Ensure your root directory contains a `Datasets` folder with the following files:
- `mercarci_dataset_cleaned_test.csv` (Training data)
- `brand_data.json`, `size_data.json`, `color_data.json` (ID mappings)
- `main_categories.json`, `sub_categories.json` (Category mappings)

2. Initializing & Training the Model
You can pre-train the models and save the `.joblib` weights to your disk by running the connection script:

bash: 
python ConnectModel.py

Dataset Information:

This project uses data from:

Data Source:
MerRec: A Large-scale Multipurpose Mercari Dataset for Consumer-to-Consumer Recommendation Systems

Authors: Lichi Li et al.
Source: https://huggingface.co/datasets/mercari-us/merrec

Licensed under CC BY-NC 4.0
