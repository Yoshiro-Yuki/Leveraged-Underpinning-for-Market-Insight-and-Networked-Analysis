import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

class ClickstreamAnalysis:
    """Clickstream model trainer and evaluator for attraction, interest, and conversion predictions."""

    def __init__(self):
        """Initialize dataset and LightGBM parameters.

        Attributes:
            data (pd.DataFrame): raw clickstream data.
            params (dict): default LightGBM classifier parameters.
        """
        self.data = pd.read_csv('..\\Datasets\\mercarci_dataset_cleaned_test.csv')
        self.params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'n_jobs': -1,
            'is_unbalance': True,
        }

    def _string_to_number(self, test_df=None):
        """Extracts numerical features from string columns.
        
        Args:
            test_df (pd.DataFrame, optional): Dataframe to process. Defaults to None.

        Returns:
            pd.DataFrame or None: Processed dataframe or None if operating on self.df.
        """
        if test_df is not None:
            test_df['name_len'] = test_df['name'].apply(len)
            test_df['num_digits'] = test_df['name'].apply(lambda x: sum(s.isdigit() for s in x ))
            test_df['upper_ratio'] = test_df['name'].apply(lambda x: sum(1 for s in x if s.isupper())/(len(x)+1))

            return test_df
        else:
            self.data['name_len'] = self.data['name'].apply(len)
            self.data['num_digits'] = self.data['name'].apply(lambda x: sum(s.isdigit() for s in x ))
            self.data['upper_ratio'] = self.data['name'].apply(lambda x: sum(1 for s in x if s.isupper())/(len(x)+1))
            
            return None

    def _preprocess_data(self):
        """Preprocess raw clickstream data for modeling.

        This method selects model features, computes category average price, one-hot encodes event types,
        and adds binary interest/conversion target columns.
        """
        self.data = self.data[['name', 'c0_name', 'price', 'brand_name_id', 'item_condition_id', 'shipper_id', 'color_id', 'size_id', 'event_id', 'name_len', 'num_digits', 'upper_ratio']]

        global_avg = self.data['price'].mean()
        self.data['cat_avg'] = self.data['c0_name'].map(self.data.groupby('c0_name')['price'].mean())

        self.data = pd.get_dummies(self.data, columns=['event_id'], prefix='action', dtype='int')

        self.data['action_interest'] = self.data[['action_item_like', 'action_item_add_to_cart_tap']].max(axis=1)
        self.data['action_conversion'] = self.data[['action_buy_comp', 'action_buy_start', 'action_offer_make']].max(axis=1)

    def train_model(self):
        """Train three LightGBM models for attraction, interest, and conversion.

        The method prepares features, splits into train/test sets, and fits separate binary classifiers.
        """
        self._string_to_number()
        self._preprocess_data()

        X = self.data[['price', 'item_condition_id', 'shipper_id', 'brand_name_id', 'color_id', 'size_id', 'cat_avg', 'name_len', 'num_digits', 'upper_ratio']]
        y = self.data.drop(['price', 'item_condition_id', 'shipper_id', 'brand_name_id', 'color_id', 'size_id', 'cat_avg', 'name_len', 'num_digits', 'upper_ratio'], axis=1)

        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=67)

        self.model_attraction = lgb.LGBMClassifier(**self.params).fit(X_train, y_train['action_item_view'])
        self.model_interest = lgb.LGBMClassifier(**self.params).fit(X_train, y_train['action_interest'])
        self.model_conversion = lgb.LGBMClassifier(**self.params).fit(X_train, y_train['action_conversion'])

    def get_predictions(self, name, price, item_condition, shipper, category, brand_name_id=14765, color_id=2998, size_id=0):
        """Predict attraction, interest, and conversion for a single listing.

        Args:
            name (str): Listing name.
            price (float): Item price.
            item_condition (int): Condition code.
            shipper (int): Shipper code.
            category (str): Category code used for category average price.
            brand_name (str): Brand name.
            color_id (int, optional): Color ID. Defaults to 2998.
            size_id (int, optional): Size ID. Defaults to 0.
        """
        test_data = pd.DataFrame(
            [[name, price, item_condition, shipper, brand_name_id, color_id, size_id]],
            columns=['name', 'price', 'item_condition_id', 'shipper_id', 'brand_name_id', 'color_id', 'size_id'],
        )

        test_data = self._string_to_number(test_data)
        test_data['cat_avg'] = self.data.loc[
            self.data['c0_name'] == category, 'price'
        ].mean()

        feature_cols = ['price', 'item_condition_id', 'shipper_id', 'brand_name_id', 
                        'color_id', 'size_id', 'cat_avg', 'name_len', 'num_digits', 'upper_ratio']
        
        test_data_final = test_data[feature_cols]

        m1 = self.model_attraction.predict_proba(test_data_final)[0][1] * 100
        m2 = self.model_interest.predict_proba(test_data_final)[0][1] * 100
        m3 = self.model_conversion.predict_proba(test_data_final)[0][1] * 100

        return round(m1, 2), round(m2, 2), round(m3, 2)

    def get_model_score(self):
        """Print validation metrics for all trained clickstream models.

        Metrics include accuracy, ROC AUC, and classification report for each target.
        """
        targets = {
            'Attraction': ('action_item_view', self.model_attraction),
            'Interest': ('action_interest', self.model_interest),
            'Conversion': ('action_conversion', self.model_conversion),
        }

        for name, (col, model) in targets.items():
            y_true = self.y_test[col]
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1]

            print(f"\n=== {name} Model Validation ===")
            print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
            print(f"ROC AUC:  {roc_auc_score(y_true, y_prob):.4f}")
            print("Classification report:")
            print(classification_report(y_true, y_pred, zero_division=0))