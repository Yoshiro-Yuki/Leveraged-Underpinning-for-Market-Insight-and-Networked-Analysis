import joblib
import os
import numpy as np
from pathlib import Path

# Assuming these modules are in your local directory/path
from PurchaseMotivation import PriceSuggestor
from ClickstreamAnalysis_v2 import ClickstreamAnalysis

class MetaModel:
    def __init__(self):
        """
        Orchestrator for the entire pipeline.
        Combines Pricing Intelligence with Funnel Decision Logic.
        """
        self.price_suggestor = PriceSuggestor()
        self.clickstream_model = ClickstreamAnalysis()

    def train_models(self):
        """Trains both the pricing engine and the funnel classifiers."""
        print("--- Training Meta-Model Pipeline ---")
        print("1/2: Training Price Suggestor...")
        self.price_suggestor.train_models()
        print("2/2: Training Clickstream Analysis...")
        self.clickstream_model.train_model()
        print("All sub-models trained successfully.")

    def predict(self, name, brand_name, brand_id, item_condition, shipper, 
                category_0, category_2, color_id, size_id, 
                price_override=None, researched_price=None):
        """
        Executes the full prediction flow.
        Returns: (Suggested Price, Probabilities, Decisions, Priority Scores)
        """
        
        # 1. PRICING LAYER
        if price_override is None:
            price_val = self.price_suggestor.predict_product(
                name=name, brand_name=brand_name, category=category_2, 
                shipper=shipper, item_condition=item_condition, 
                color_id=color_id, size_id=size_id
            )
        else:
            try:
                price_val = float(price_override)
            except (ValueError, TypeError):
                price_val = 0.0

        # 2. CLICKSTREAM LAYER (The 3-Layer Decision System)
        # ensure get_predictions returns lists for [probs], [decisions], [priority]
        probs, decisions, priority = self.clickstream_model.get_predictions(
            name=name, 
            price=price_val, 
            item_condition=item_condition, 
            shipper=shipper, 
            category=category_0, 
            brand_name_id=brand_id, 
            color_id=color_id, 
            size_id=size_id, 
            researched_price=researched_price
        )

        return {
            'active_price': round(price_val, 2),
            'attraction': {'prob': probs[0], 'decision': decisions[0], 'intent': priority[0]},
            'interest':   {'prob': probs[1], 'decision': decisions[1], 'intent': priority[1]},
            'conversion': {'prob': probs[2], 'decision': decisions[2], 'intent': priority[2]}
        }

    def save_model(self, filepath=None):
        """Saves the entire pipeline state using cross-platform paths."""
        if filepath is None:
            filepath = Path("..") / "Datasets" / "meta_model_v1_2.joblib"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(self, filepath, compress=3)
        print(f"✅ Meta-Model saved to {os.path.abspath(filepath)}")

    @classmethod
    def load_model(cls, filepath=None):
        """Loads a pre-trained meta-model or starts training if not found."""
        if filepath is None:
            filepath = Path("..") / "Datasets" / "meta_model_v1_2.joblib"
        else:
            filepath = Path(filepath)

        if filepath.exists():
            print(f"📦 Loading Meta-Model from {filepath}...")
            return joblib.load(filepath)
        else:
            print("⚠️ No saved model found. Initializing new training session...")
            new_model = cls()
            new_model.train_models()
            new_model.save_model(filepath)
            return new_model

if __name__ == "__main__":
    # Initialize Path
    MODEL_PATH = Path("..") / "Datasets" / "meta_model_v1_2.joblib"
    
    # Load or Train
    meta = MetaModel.load_model(MODEL_PATH)
    
    # Test a prediction
    result = meta.predict(
        name="Vintage Denim Jacket", 
        brand_name="Levi's", 
        brand_id=123, 
        item_condition=2, 
        shipper=1, 
        category_0="Clothing", 
        category_2="Outerwear", 
        color_id=2298, 
        size_id=0,
        researched_price=85.0 
    )
    
    if result:
        print("\n" + "="*40)
        print(f"FINAL REPORT: {result['active_price']} $")
        print("="*40)
        for stage in ['attraction', 'interest', 'conversion']:
            data = result[stage]
            status = "🟢 ACT" if data['decision'] == 1 else "🔴 WAIT"
            # Formatting to handle potential float/int inputs
            prob_str = f"{data['prob']:.2f}%" if isinstance(data['prob'], (float, int)) else data['prob']
            print(f"{stage.upper():10} | {status} | Intent: {data['intent']}/100 | Prob: {prob_str}")