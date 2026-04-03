import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, classification_report, confusion_matrix
from imblearn.ensemble import BalancedRandomForestClassifier

class ClickstreamAnalysis:
    def __init__(self):

        self.raw_data = pd.read_csv('..\\Datasets\\mercarci_dataset_cleaned_test.csv')

        # F-Beta Parameters: Used to 'penalize' stages where the model is historically weak.
        # Format: (Precision, Recall, Beta)
        self.perf_params = {
            'attraction': (0.23, 0.31, 0.5),
            'interest':   (0.25, 0.17, 0.5),
            'conversion': (0.1, 0.33, 0.4)
        }

        # DECISION LAYER: The 'Line in the Sand' for taking action (ACT vs WAIT).
        self.thresholds = {'attraction': 0.4, 'interest': 0.6, 'conversion': 0.8}
        
        # PRIORITY LAYER: Multipliers to amplify small probabilities into 0-100 scores.
        self.rewards = {'attraction': 3.0, 'interest': 1.8, 'conversion': 7.0}

        self.lgb_params = {
            'objective': 'binary', 
            'metric': 'binary_logloss',
            'verbosity': -1, 
            'n_jobs': -1, 
            'is_unbalance': True, # Crucial for Mercari's imbalanced 'Buy' events
        }

        self.feature_cols = [
            'price', 'item_condition_id', 'shipper_id', 'brand_name_id', 
            'color_id', 'size_id', 'cat_avg', 'price_delta', 
            'researched_delta', 'has_research', 'name_len', 'num_digits', 'upper_ratio'
        ]

    def _string_to_number(self, df: pd.DataFrame) -> pd.DataFrame:
        """FEATURE ENGINEERING: Vectorized string operations for better performance."""
        df['name_len'] = df['name'].str.len()
        df['num_digits'] = df['name'].str.count(r'\d')
        # Calculates what percentage of the title is SHOUTING (Uppercase)
        df['upper_ratio'] = df['name'].str.count(r'[A-Z]') / (df['name_len'] + 1)
        return df

    def _apply_market_context(self, df: pd.DataFrame, is_training=False):
        """
        LEAKAGE PREVENTION: We only calculate 'average category price' 
        from the training set. We never let the test set influence these means.
        """
        if is_training:
            self.cat_means = df.groupby('c0_name')['price'].mean()
            self.global_mean = df['price'].mean() # Fallback for new categories
        
        # Map learned means back to the current dataframe
        df['cat_avg'] = df['c0_name'].map(self.cat_means).fillna(self.global_mean)
        df['price_delta'] = df['price'] - df['cat_avg']
        
        if 'researched_price' not in df.columns:
            df['researched_price'] = np.nan
            
        df['has_research'] = df['researched_price'].notna().astype(int)
        df['researched_delta'] = (df['price'] - df['researched_price']).fillna(0)
        return df

    def train_model(self):
        """TRAINING PIPELINE: Splits data first, then trains the 3-stage funnel."""
        df = self._string_to_number(self.raw_data)
        
        # Convert event IDs into binary targets for the funnel
        df = pd.get_dummies(df, columns=['event_id'], prefix='action', dtype='int')
        df['action_interest'] = df[['action_item_like', 'action_item_add_to_cart_tap']].max(axis=1)
        df['action_conversion'] = df[['action_buy_comp', 'action_buy_start', 'action_offer_make']].max(axis=1)
        
        # THE SPLIT: Must happen BEFORE market context to ensure 0% data leakage.
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=67)
        
        train_df = self._apply_market_context(train_df, is_training=True)
        self.processed_test = self._apply_market_context(test_df, is_training=False)
        
        self.X_test = self.processed_test[self.feature_cols]
        self.y_test = self.processed_test[['action_item_view', 'action_interest', 'action_conversion']]

        # MODEL 1: Attraction (LightGBM)
        self.model_attraction = lgb.LGBMClassifier(**self.lgb_params).fit(
            train_df[self.feature_cols], train_df['action_item_view'])
        
        # MODEL 2: Interest (LightGBM)
        self.model_interest = lgb.LGBMClassifier(**self.lgb_params).fit(
            train_df[self.feature_cols], train_df['action_interest'])
        
        # MODEL 3: Conversion (Balanced Random Forest for high-imbalance precision)
        self.model_conversion = BalancedRandomForestClassifier(
            n_estimators=100, random_state=67, n_jobs=-1).fit(
            train_df[self.feature_cols], train_df['action_conversion'])

    def _penalty_factor(self, stage):
        """MATH: Calculates the F-Beta trust score based on historical performance."""
        p, r, b = self.perf_params[stage]
        return (1 + b**2) * (p * r) / ((b**2 * p) + r)

    def soft_threshold(self, prob, threshold, steepness=8):
        """MATH: Sigmoid shaping to amplify differences around the decision threshold."""
        return 1 / (1 + np.exp(-steepness * (prob - threshold)))

    def get_predictions(self, name, price, item_condition, shipper, category, **kwargs):
        """
        THE 3-LAYER OUTPUT:
        Layer 1: Probabilities (%) - Statistical Truth
        Layer 2: Decisions (1/0) - Actionable Binary
        Layer 3: Priority Index (0-100) - Intent Ranking
        """
        # Prepare single-row dataframe for prediction
        test_data = pd.DataFrame([[name, price, item_condition, shipper, category]], 
                                columns=['name', 'price', 'item_condition_id', 'shipper_id', 'c0_name'])
        
        for col in ['brand_name_id', 'color_id', 'size_id']:
            test_data[col] = kwargs.get(col, 0)
        
        test_data = self._string_to_number(test_data)
        test_data['researched_price'] = kwargs.get('researched_price', np.nan)
        test_data = self._apply_market_context(test_data, is_training=False)
        
        # LAYER 1: RAW PROBABILITIES
        probs = [
            self.model_attraction.predict_proba(test_data[self.feature_cols])[0][1],
            self.model_interest.predict_proba(test_data[self.feature_cols])[0][1],
            self.model_conversion.predict_proba(test_data[self.feature_cols])[0][1]
        ]

        # LAYER 3: PRIORITY INDEX (SIGMOID TRANSFORMATION)
        final_scores = []
        stages = ['attraction', 'interest', 'conversion']
        for i, stage in enumerate(stages):
            shaped = self.soft_threshold(probs[i], self.thresholds[stage])
            trust = self._penalty_factor(stage)
            rewarded = (shaped * trust) * self.rewards[stage]
            final_scores.append(min(100.0, rewarded * 100))

        # LAYER 2: DECISIONS (BINARY)
        decision_binary = [1 if p >= self.thresholds[stages[i]] else 0 for i, p in enumerate(probs)]
        
        probability_pct = [round(p * 100, 2) for p in probs]
        priority_index = tuple(round(s, 2) for s in final_scores)
        
        return probability_pct, decision_binary, priority_index

    def validate_decisions(self):
        """VALIDATION: Evaluate the Decision Layer using Confusion Matrices."""
        stages = ['attraction', 'interest', 'conversion']
        targets = ['action_item_view', 'action_interest', 'action_conversion']
        models = [self.model_attraction, self.model_interest, self.model_conversion]
        
        print("\n" + "="*50)
        print("DECISION LAYER VALIDATION (Confusion Matrices)")
        print("="*50)

        for i, stage in enumerate(stages):
            raw_p = models[i].predict_proba(self.X_test)[:, 1]
            y_pred = (raw_p >= self.thresholds[stage]).astype(int) # Apply threshold
            y_true = self.y_test[targets[i]]

            cm = confusion_matrix(y_true, y_pred)
            print(f"STAGE: {stage.upper()} | Threshold: {self.thresholds[stage]}")
            print(f"Confusion Matrix (TN, FP, FN, TP):\n{cm}")
            # Precision/Recall report helps find if threshold is too strict or too loose
            print(classification_report(y_true, y_pred, target_names=['WAIT', 'ACT'], zero_division=0))
            print("-" * 30)

if __name__ == "__main__":
    # RUNNER: Initialize, Train, Validate, and Predict
    ca = ClickstreamAnalysis()
    ca.train_model()
    ca.validate_decisions()
    
    # Final check with a dummy item
    prob, dec, prio = ca.get_predictions(
        name="Vintage Leather Jacket", price=79.99, item_condition=2, shipper=1, 
        category='Clothing', researched_price=85.00
    )
    
    print("\nExample Prediction (Vintage Jacket):")
    print(f" -> Probabilities: {prob}")
    print(f" -> Decisions:     {dec}")
    print(f" -> Priority Index:{prio}")