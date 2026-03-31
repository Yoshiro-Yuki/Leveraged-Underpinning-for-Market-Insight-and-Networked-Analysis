import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.ensemble import BalancedRandomForestClassifier

class ClickstreamAnalysis:
    """Funnel analysis model using hybrid LightGBM and Balanced Random Forest.
    
    Includes an optional 'Researched Price' feature to improve conversion accuracy.
    """

    def __init__(self):
        """Initializes the analysis with dataset paths and default hyperparameters."""
        self.data = pd.read_csv('..\\Datasets\\mercarci_dataset_cleaned_test.csv')
        self.lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'n_jobs': -1,
            'is_unbalance': True,
        }
        # Added new features to the core list
        self.feature_cols = [
            'price', 'item_condition_id', 'shipper_id', 'brand_name_id', 
            'color_id', 'size_id', 'cat_avg', 'price_delta', 
            'researched_delta', 'has_research',
            'name_len', 'num_digits', 'upper_ratio'
        ]

    def _string_to_number(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts numerical text features from listing names."""
        df['name_len'] = df['name'].apply(len)
        df['num_digits'] = df['name'].apply(lambda x: sum(s.isdigit() for s in x))
        df['upper_ratio'] = df['name'].apply(
            lambda x: sum(1 for s in x if s.isupper()) / (len(x) + 1)
        )
        return df

    def _preprocess_data(self):
        """Executes the full preprocessing pipeline with optional research logic."""
        self.data = self._string_to_number(self.data)
        
        # Market Context Engineering
        self.cat_means = self.data.groupby('c0_name')['price'].mean()
        self.data['cat_avg'] = self.data['c0_name'].map(self.cat_means)
        self.data['price_delta'] = self.data['price'] - self.data['cat_avg']
        
        # Optional Research Price Logic (Initializes columns if not present in CSV)
        if 'researched_price' not in self.data.columns:
            self.data['researched_price'] = np.nan
            
        self.data['has_research'] = self.data['researched_price'].notna().astype(int)
        self.data['researched_delta'] = (self.data['price'] - self.data['researched_price']).fillna(0)
        
        # Event Encoding & Target Aggregation
        self.data = pd.get_dummies(self.data, columns=['event_id'], prefix='action', dtype='int')
        self.data['action_interest'] = self.data[['action_item_like', 'action_item_add_to_cart_tap']].max(axis=1)
        self.data['action_conversion'] = self.data[['action_buy_comp', 'action_buy_start', 'action_offer_make']].max(axis=1)

    def train_model(self):
        """Trains stage-specific classifiers for the clickstream funnel."""
        self._preprocess_data()
        X = self.data[self.feature_cols]
        y = self.data[['action_item_view', 'action_interest', 'action_conversion']]
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=67)

        print("Training Attraction & Interest models...")
        self.model_attraction = lgb.LGBMClassifier(**self.lgb_params).fit(X_train, y_train['action_item_view'])
        self.model_interest = lgb.LGBMClassifier(**self.lgb_params).fit(X_train, y_train['action_interest'])

        print("Training Conversion model (Balanced Random Forest)...")
        self.model_conversion = BalancedRandomForestClassifier(
            n_estimators=100, sampling_strategy='auto', replacement=True, random_state=67, n_jobs=-1
        ).fit(X_train, y_train['action_conversion'])

    def get_predictions(self, name, price, item_condition, shipper, category, 
                        brand_name_id=14765, color_id=2998, size_id=0, researched_price=None):
        """Predicts funnel probabilities, incorporating optional researched price.

        Args:
            researched_price (float, optional): External researched market value.
        """
        test_data = pd.DataFrame(
            [[name, price, item_condition, shipper, brand_name_id, color_id, size_id]],
            columns=['name', 'price', 'item_condition_id', 'shipper_id', 'brand_name_id', 'color_id', 'size_id'],
        )
        test_data = self._string_to_number(test_data)
        
        # Feature Engineering for single row
        cat_avg = self.cat_means.get(category, self.data['price'].mean())
        test_data['cat_avg'] = cat_avg
        test_data['price_delta'] = price - cat_avg
        
        # Handle Optional Research
        test_data['has_research'] = 1 if researched_price is not None else 0
        test_data['researched_delta'] = (price - researched_price) if researched_price is not None else 0
        
        final_input = test_data[self.feature_cols]
        m1 = self.model_attraction.predict_proba(final_input)[0][1] * 100
        m2 = self.model_interest.predict_proba(final_input)[0][1] * 100
        m3 = self.model_conversion.predict_proba(final_input)[0][1] * 100
        return round(m1, 2), round(m2, 2), round(m3, 2)
    
    def get_model_score(self):
        """Evaluates all three funnel stages and returns a summary of performance."""
        if not hasattr(self, 'X_test'):
            return "Error: Model must be trained before scoring."

        stages = {
            "Attraction (View)": (self.model_attraction, self.y_test['action_item_view']),
            "Interest (Like/Cart)": (self.model_interest, self.y_test['action_interest']),
            "Conversion (Buy/Offer)": (self.model_conversion, self.y_test['action_conversion'])
        }

        results = []
        results.append("=== Model Performance Evaluation ===\n")

        for name, (model, y_true) in stages.items():
            # Get probability scores for AUC
            y_probs = model.predict_proba(self.X_test)[:, 1]
            # Get hard predictions for Classification Report
            y_pred = model.predict(self.X_test)
            
            auc = roc_auc_score(y_true, y_probs)
            report = classification_report(y_true, y_pred)
            
            results.append(f"Stage: {name}")
            results.append(f"ROC-AUC Score: {auc:.4f}")
            results.append("Classification Report:")
            results.append(report)
            results.append("-" * 40)

        return "\n".join(results)

if __name__ == "__main__":
    ca = ClickstreamAnalysis()
    ca.train_model()
    print(ca.get_model_score())