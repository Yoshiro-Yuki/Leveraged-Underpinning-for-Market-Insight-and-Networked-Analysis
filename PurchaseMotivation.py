import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb

class PriceSuggestor:
    """A price suggestion engine using a stacked ensemble model.
    
    Attributes:
        df (pd.DataFrame): The loaded dataset.
        random_seed (int): Seed for reproducibility.
        lg_cols (list): Features used specifically for the LightGBM model.
    """

    def __init__(self, path='..\\Datasets\\mercarci_dataset_cleaned_test.csv'):
        """Initializes the suggestor with data from the given path."""
        self.df = pd.read_csv(path) 
        self.random_seed = 67
        self.lg_cols = ['brand_name_trg', 'c2_name_trg', 'item_condition_id', 'color_id', 'size_id', 'shipper_id', 'name_len', 'num_digits', 'upper_ratio']

    def _split_dataset(self):
        """Splits the data into training, meta-training, and test sets.
        
        Returns:
            None
        """
        self.df = self.df.drop(['price','session_length_min'], axis=1)
        
        self.X, self.y = self.df.drop('log_price', axis=1), self.df['log_price']

        self.X_train, self.X_temp, self.y_train, self.y_temp = train_test_split(self.X, self.y, test_size=0.4, random_state=self.random_seed)
        self.X_meta, self.X_test, self.y_meta, self.y_test = train_test_split(self.X_temp, self.y_temp, test_size=0.5, random_state=self.random_seed)

        return None

    def _oof_target_encoding(self, target_column):
        """Performs Out-Of-Fold target encoding to prevent data leakage.
        
        Args:
            target_column (str): The name of the column to encode.

        Returns:
            None
        """
        self.X_train = self.X_train.copy()
        self.X_train['temp_target_price'] = self.y_train

        # Create column placeholder
        self.X_meta[target_column + '_trg'] = np.nan
        self.X_test[target_column + '_trg'] = np.nan
    
        # Instance kfold
        kf = KFold(n_splits=5,shuffle=True,random_state=self.random_seed)
    
        # Blinded inference on other splits
        for t_idx, v_idx in kf.split(self.X_train):
            mean_map = self.X_train.iloc[t_idx].groupby(target_column)['temp_target_price'].mean()
            self.X_train.loc[self.X_train.index[v_idx], target_column + '_trg'] = self.X_train.loc[self.X_train.index[v_idx], target_column].map(mean_map)
    
        # Applying on the meta and test dataframe
        global_map = self.X_train.groupby(target_column)['temp_target_price'].mean()
        self.X_meta[target_column + '_trg'] = self.X_meta[target_column].map(global_map)
        self.X_test[target_column + '_trg'] = self.X_test[target_column].map(global_map)
    
        # Fill all null with global mean
        g_mean = self.y_train.mean()
        for df in [self.X_train, self.X_meta, self.X_test]:
            df[target_column + '_trg'] = df[target_column + '_trg'].fillna(g_mean)
            
        return None

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
            self.df['name_len'] = self.df['name'].apply(len)
            self.df['num_digits'] = self.df['name'].apply(lambda x: sum(s.isdigit() for s in x ))
            self.df['upper_ratio'] = self.df['name'].apply(lambda x: sum(1 for s in x if s.isupper())/(len(x)+1))
            
            return None

    def _train_agents(self):
        """Trains the base learners (Ridge and LightGBM).
        
        Returns:
            None
        """
        self._string_to_number()
        self._split_dataset()
        self._oof_target_encoding('brand_name')
        self._oof_target_encoding('c2_name')

        self.tv = TfidfVectorizer(
            max_features=25000,
            ngram_range=(1,2))
        
        self.r_ridge = Ridge(
            alpha=0.1)
        
        self.r_lgb = lgb.LGBMRegressor(
            n_estimators=1000, 
            num_leaves=80, 
            verbosity=-1)

        # Text Vectorization
        self.X_txt_train = self.tv.fit_transform(self.X_train['name'])
        
        # Ridge regression
        self.m_ridge = self.r_ridge.fit(self.X_txt_train, self.y_train)
        
        # LGBM 
        self.m_lgbm = self.r_lgb.fit(self.X_train[self.lg_cols],self.y_train)

        return None

    def _get_meta_values(self, df_data):
        """Generates predictions from base learners to be used as meta-features.
        
        Args:
            df_data (pd.DataFrame): The input data.

        Returns:
            np.ndarray: Stacked features for the meta-model.
        """
        p1 = self.r_ridge.predict(self.tv.transform(df_data['name']))
        p2 = self.r_lgb.predict(df_data[self.lg_cols])

        return np.column_stack([p1, p2, df_data[['shipper_id', 'item_condition_id']].values])

    def train_models(self):
        """Trains the final meta-model using base model outputs.
        
        Returns:
            None
        """
        self._train_agents()

        self.meta_model = Ridge(alpha=0.01).fit(
            self._get_meta_values(self.X_meta),
            self.y_meta)

        return None

    def model_score(self):
        """Calculates and returns the R2 score of the ensemble model.
        
        Returns:
            str: A formatted string containing the R2 score.
        """
        self.y_true = np.expm1(self.y_test)
        self.y_pred = np.expm1(self.meta_model.predict(self._get_meta_values(self.X_test)))
        
        print(f"Final R2: {r2_score(self.y_test, self.meta_model.predict(self._get_meta_values(self.X_test)))}")
        print(f"Final MAE: {mean_absolute_error(self.y_true, self.y_pred)}")
    

    def predict_product(self, name, brand_name, category, shipper, item_condition, color_id=2298, size_id=0):
        """Predicts the price for a single product.
        
        Args:
            name (str): Item name.
            brand_name (str): Brand of the item.
            category (str): Sub-category name.
            shipper (int): Shipper ID.
            item_condition (int): Condition ID.
            color (int): Color ID of the item.
            size_id (int): Size ID.

        Returns:
            int: Predicted price rounded to 2 decimal places.
        """
        predict_prod = pd.DataFrame({
            'name':[name],
            'brand_name':[brand_name],
            'c2_name': [category],
            'shipper_id':[shipper],
            'item_condition_id':[item_condition],
            'color_id': [color_id],
            'size_id': [size_id]
        })

        brand_map = self.X_train.groupby('brand_name')['temp_target_price'].mean()
        categ_map = self.X_train.groupby('c2_name')['temp_target_price'].mean()

        predict_prod['brand_name_trg'] = predict_prod['brand_name'].map(brand_map).fillna(self.y_train.mean())
        predict_prod['c2_name_trg'] = predict_prod['c2_name'].map(categ_map).fillna(self.y_train.mean())
        
        predict_prod = self._string_to_number(test_df=predict_prod)
        prediction = self.meta_model.predict(self._get_meta_values(predict_prod))

        return round(np.expm1(prediction[0]), 2)
    
if __name__ == "__main__":
    suggestor = PriceSuggestor()
    suggestor.train_models()
    print(suggestor.model_score())