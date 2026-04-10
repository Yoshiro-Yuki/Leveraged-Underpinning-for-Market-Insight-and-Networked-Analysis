import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve 
from imblearn.ensemble import BalancedRandomForestClassifier

class ClickstreamAnalysis:
    def __init__(self):
        self.thresholds = {'attraction': 0.4, 'interest': 0.6, 'conversion': 0.8}
        self.rewards = {'attraction': 4.5, 'interest': 1.8, 'conversion': 7.0}
        self.calibrators = {} 
        
        self.lgb_params = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'verbosity': -1, 'n_jobs': -1, 'is_unbalance': True, 
        }

        self.feature_cols = [
            'price', 'item_condition_id', 'shipper_id', 'brand_name_id', 
            'color_id', 'size_id', 'cat_avg', 'price_delta', 
            'researched_delta', 'has_research', 'name_len', 'num_digits', 'upper_ratio'
        ]

    def _string_to_number(self, df: pd.DataFrame) -> pd.DataFrame:
        df['name_len'] = df['name'].str.len()
        df['num_digits'] = df['name'].str.count(r'\d')
        df['upper_ratio'] = df['name'].str.count(r'[A-Z]') / (df['name_len'] + 1)
        return df

    def _apply_market_context(self, df: pd.DataFrame, is_training=False):
        if is_training:
            self.cat_means = df.groupby('c0_name')['price'].mean()
            self.global_mean = df['price'].mean()
        
        df['cat_avg'] = df['c0_name'].map(self.cat_means).fillna(self.global_mean)
        df['price_delta'] = df['price'] - df['cat_avg']
        if 'researched_price' not in df.columns:
            df['researched_price'] = np.nan
        df['has_research'] = df['researched_price'].notna().astype(int)
        df['researched_delta'] = (df['price'] - df['researched_price']).fillna(0)
        return df

    def sigmoid_smoothing(self, prob, threshold, steepness=8):
        return 1 / (1 + np.exp(-steepness * (prob - threshold)))

    def train_model(self, data_path):
        df = pd.read_csv(data_path)
        df = self._string_to_number(df)
        df = pd.get_dummies(df, columns=['event_id'], prefix='action', dtype='int')
        
        df['action_interest'] = df[['action_item_like', 'action_item_add_to_cart_tap']].max(axis=1)
        df['action_conversion'] = df.get('action_buy_comp', 0) 

        train_df, temp_df = train_test_split(df, test_size=0.3, shuffle=False)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, shuffle=False)
        sig_train_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
        
        train_df = self._apply_market_context(train_df, is_training=True)
        val_df = self._apply_market_context(val_df, is_training=False)
        sig_train_df = self._apply_market_context(sig_train_df, is_training=False)
        self.processed_test = self._apply_market_context(test_df, is_training=False)
        
        self.X_test = self.processed_test[self.feature_cols]
        self.y_test = self.processed_test[['action_item_view', 'action_interest', 'action_conversion']]

        # 1. ATTRACTION & 2. INTEREST
        self.model_attraction = lgb.LGBMClassifier(**self.lgb_params).fit(
            train_df[self.feature_cols], train_df['action_item_view'])
        
        self.model_interest = lgb.LGBMClassifier(**self.lgb_params).fit(
            train_df[self.feature_cols], train_df['action_interest'])
        
        # Isotonic Calibration Loop
        configs = [
            {'target': 'action_item_view', 'name': 'attraction', 'model': self.model_attraction},
            {'target': 'action_interest', 'name': 'interest', 'model': self.model_interest}
        ]
        
        for cfg in configs:
            raw_probs = cfg['model'].predict_proba(val_df[self.feature_cols])[:, 1]
            iso_reg = IsotonicRegression(out_of_bounds='clip')
            iso_reg.fit(raw_probs, val_df[cfg['target']])
            self.calibrators[cfg['name']] = iso_reg
        
        # 3. CONVERSION
        self.model_conversion = BalancedRandomForestClassifier(
            n_estimators=300, random_state=67, sampling_strategy=0.5, n_jobs=-1).fit(
            sig_train_df[self.feature_cols], sig_train_df['action_conversion'])

    def get_predictions(self, name, price, item_condition, shipper, category, **kwargs):
        test_data = pd.DataFrame([[name, price, item_condition, shipper, category]], 
                                columns=['name', 'price', 'item_condition_id', 'shipper_id', 'c0_name'])
        for col in ['brand_name_id', 'color_id', 'size_id']:
            test_data[col] = kwargs.get(col, 0)
        
        test_data = self._string_to_number(test_data)
        test_data['researched_price'] = kwargs.get('researched_price', np.nan)
        test_data = self._apply_market_context(test_data, is_training=False)
        
        # Raw Probs
        p_attr_raw = self.model_attraction.predict_proba(test_data[self.feature_cols])[0][1]
        p_int_raw = self.model_interest.predict_proba(test_data[self.feature_cols])[0][1]
        p_conv_raw = self.model_conversion.predict_proba(test_data[self.feature_cols])[0][1]

        # 1. Isotonic Calibration
        s_attr_cal = self.calibrators['attraction'].transform([p_attr_raw])[0]
        s_int_cal = self.calibrators['interest'].transform([p_int_raw])[0]

        # 2. Sigmoid Ranking (using calibrated inputs for attr/int)
        s_attr = self.sigmoid_smoothing(s_attr_cal, self.thresholds['attraction'])
        s_int = self.sigmoid_smoothing(s_int_cal, self.thresholds['interest'])
        s_conv = self.sigmoid_smoothing(p_conv_raw, self.thresholds['conversion'])

        probs_calibrated_pct = [round(p * 100, 2) for p in [s_attr_cal, s_int_cal, p_conv_raw]]
        decisions = [1 if p >= self.thresholds[s] else 0 for s, p in zip(['attraction', 'interest', 'conversion'], [s_attr, s_int, s_conv])]
        shaped_scores = (round(s_attr, 2), round(s_int, 2), round(s_conv, 2))
        
        return probs_calibrated_pct, decisions, shaped_scores

    def validate_decisions(self):
        print("\n" + "="*65)
        print("FINAL HOLDOUT VALIDATION (Metrics on Calibrated Probs)")
        print("="*65)

        p_attr_raw = self.model_attraction.predict_proba(self.X_test)[:, 1]
        p_attr_cal = self.calibrators['attraction'].transform(p_attr_raw)
        p_attr_adj = self.sigmoid_smoothing(p_attr_cal, self.thresholds['attraction'])
        
        p_int_raw = self.model_interest.predict_proba(self.X_test)[:, 1]
        p_int_cal = self.calibrators['interest'].transform(p_int_raw)
        p_int_adj = self.sigmoid_smoothing(p_int_cal, self.thresholds['interest'])
        
        print(f"STAGE: ATTRACTION | Brier: {brier_score_loss(self.y_test['action_item_view'], p_attr_cal):.4f} | AUC: {roc_auc_score(self.y_test['action_item_view'], p_attr_adj):.4f}")
        print(f"STAGE: INTEREST   | Brier: {brier_score_loss(self.y_test['action_interest'], p_int_cal):.4f} | AUC: {roc_auc_score(self.y_test['action_interest'], p_int_adj):.4f}")
        
        p_conv = self.model_conversion.predict_proba(self.X_test)[:, 1]
        p_conv_adj = self.sigmoid_smoothing(p_conv, self.thresholds['conversion'])
        print(f"STAGE: CONVERSION | Brier: {brier_score_loss(self.y_test['action_conversion'], p_conv):.4f} | AUC: {roc_auc_score(self.y_test['action_conversion'], p_conv_adj):.4f}")

    def plot_performance(self):
        stages = [
            {'name': 'attraction', 'target': 'action_item_view', 'model': self.model_attraction},
            {'name': 'interest', 'target': 'action_interest', 'model': self.model_interest}
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for i, stage in enumerate(stages):
            # Dynamic model selection fix
            p_raw = stage['model'].predict_proba(self.X_test)[:, 1]
            p_calibrated = self.calibrators[stage['name']].transform(p_raw)
            p_adjusted = self.sigmoid_smoothing(p_calibrated, self.thresholds[stage['name']])
            y_true = self.y_test[stage['target']]
            
            # ROC (Ranking)
            fpr, tpr, _ = roc_curve(y_true, p_calibrated)
            axes[i, 0].plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc_score(y_true, p_adjusted):.4f}')
            axes[i, 0].plot([0, 1], [0, 1], linestyle='--', color='gray')
            axes[i, 0].set_title(f"{stage['name'].upper()} ROC (Calibrated)")
            axes[i, 0].legend()

            # Calibration (Reliability)
            prob_true, prob_pred = calibration_curve(y_true, p_calibrated, n_bins=10)
            axes[i, 1].plot(prob_pred, prob_true, marker='s', color='green', label='Isotonic')
            axes[i, 1].plot([0, 1], [0, 1], linestyle='--', color='gray')
            axes[i, 1].set_title(f"{stage['name'].upper()} Calibration Curve")
            axes[i, 1].legend()

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    ca = ClickstreamAnalysis()
    ca.train_model('..\\Datasets\\mercarci_dataset_cleaned_test.csv') 
    ca.validate_decisions()
    ca.plot_performance()
    x = ca.get_predictions(
        name="Vintage Denim Jacket", 
        price=45.99, 
        item_condition=2, 
        shipper=1, 
        category="Clothing",
        brand_name_id=123,
        color_id=456,
        size_id=789,
        researched_price=50.00
    )
    print(x)