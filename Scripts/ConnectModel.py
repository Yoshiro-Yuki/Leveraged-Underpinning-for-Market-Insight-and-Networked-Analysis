import os
from meta_model import MetaModel  # This fixes the 'AttributeError'

def main():
    # Use absolute pathing to ensure the file is found regardless of terminal location
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "meta_model_v1.joblib")

    # 1. Try to load
    print("Checking for existing model...")
    mm = MetaModel.load_model(model_path)

    # 2. If not found, train and save
    if mm is None:
        print("Model file not found. Starting training (this may take a while)...")
        mm = MetaModel()
        mm.train_models()
        mm.save_model(model_path)
    else:
        print("✅ Model loaded successfully from disk.")

    return mm

def run_prediction(mm):
    # 3. Predict
    print("\n--- Running Prediction ---")
    price_res, click_res = mm.predict(
        name="iPhone 11", 
        brand_name="Apple", 
        price=500, 
        item_condition=3, 
        shipper=1, 
        category_0="Electronics", 
        category_2="Phones"
    )
    
    print(f"Suggested Price: {price_res}")
    print(f"Clickstream Scores: {click_res}")

if __name__ == "__main__":
    mm = main()
    run_prediction(mm)