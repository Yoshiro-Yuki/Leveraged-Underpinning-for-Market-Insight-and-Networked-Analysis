import joblib
import os

from PurchaseMotivation import PriceSuggestor
from ClickstreamAnalysis import ClickstreamAnalysis

class MetaModel:
    def __init__(self):
        # Initialize the classes
        self.price_suggestor = PriceSuggestor()
        self.clickstream_model = ClickstreamAnalysis()
        
        # parameters for penalty calculation
        self.precision_attraction = 0.2
        self.recall_attraction = 0.53
        self.beta_attraction = 0.5

        self.precision_interest = 0.18
        self.recall_interest = 0.53
        self.beta_interest = 0.5

        self.precision_conversion = 0.02
        self.recall_conversion = 0.56
        self.beta_conversion = 0.1

    def train_models(self):
        print("Training sub-models...")
        self.price_suggestor.train_models()
        self.clickstream_model.train_model()

    def _penalty(self, prob, precision, recall, beta):
        """Calculates a penalty based on the predicted probability and model performance."""
        p = prob / 100.0 if prob > 1.0 else prob
    
        f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-6)
        
        reliability_weight = f_beta * p

        return reliability_weight

    def predict(self, name, brand_name, brand_id, item_condition, shipper, 
                category_0, category_2, color_id, size_id, 
                price_override=None, researched_price=None):
        """Orchestrates pricing and clickstream prediction with optional research support."""
        
        # 1. Determine the price
        if price_override is None:
            price_val = self.price_suggestor.predict_product(
                name=name, brand_name=brand_name, category=category_2, 
                shipper=shipper, item_condition=item_condition, 
                color_id=color_id, size_id=size_id
            )
        else:
            price_val = float(price_override)

        # 2. Feed the active price and optional research into Clickstream model
        clickstream_pred = self.clickstream_model.get_predictions(
            name=name, price=price_val, item_condition=item_condition, 
            shipper=shipper, category=category_0, brand_name_id=brand_id, 
            color_id=color_id, size_id=size_id, 
            researched_price=researched_price # Added support
        )

        attr, inter, conv = clickstream_pred

        # Calculate score penalty for attraction probability
        penalty_attraction = self._penalty(attr, self.precision_attraction, self.recall_attraction, self.beta_attraction)
        attr = round((attr * (1 - penalty_attraction)), 2) # Adjusted attraction score

        # Calculate score penalty for interest probability
        penalty_interest = self._penalty(inter, self.precision_interest, self.recall_interest, self.beta_interest)
        inter = round((inter * (1 - penalty_interest)), 2)  # Adjusted interest score

        # Calculate score penalty for conversion probability
        penalty_conversion = self._penalty(conv, self.precision_conversion, self.recall_conversion, self.beta_conversion)
        conv = round((conv * (1 - penalty_conversion)), 2)  # Adjusted conversion score

        return price_val, (attr, inter, conv)

    def save_model(self, filepath="..\\Datasets\\meta_model_v1_2.joblib"):
        # Save the entire object state
        joblib.dump(self, filepath, compress=3)
        print(f"✅ Model saved to {os.path.abspath(filepath)}")

    @classmethod
    def load_model(cls, filepath="..\\Datasets\\meta_model_v1_2.joblib"):
        if os.path.exists(filepath):
            print(f"📦 Loading existing model from {filepath}...")
            return joblib.load(filepath)
        else:
            print("❌ No saved model found. Creating and training a new one...")
            new_model = cls()
            new_model.train_models()
            new_model.save_model(filepath)
            return new_model

if __name__ == "__main__":
    # Use the class method to either load or train/save automatically
    meta_model = MetaModel.load_model("..\\Datasets\\meta_model_v1_2.joblib")
    
    # Test prediction
    result = meta_model.predict(
        name="Vintage Denim Jacket", brand_name="Levi's", item_condition=2, shipper=1, 
        category_0="Clothing", category_2="Outerwear", color_id=2298, size_id=0
    )
    print(f"Suggested Price: {result[0]}")
    print(f"Clickstream Probabilities (Attr, Int, Conv): {result[1]}")