import pandas as pd
import numpy as np
import pickle
from typing import Optional, Any
from loguru import logger

from .utils import compute_vwap, compute_feature_eng


class InferencePipeline:
    """
    Professional inference pipeline for Kaggle submissions.
    """
    
    def __init__(self, model: Any, fast_period: int = 5, slow_period: int = 10):
        self.fast_period = fast_period
        self.slow_period = slow_period
        
        # Extract components from TrainingPipeline, ModelSelector, or raw model
        if hasattr(model, 'model_selector_'):
            self.model_selector = model.model_selector_
        elif hasattr(model, 'data_processor_'):
            self.model_selector = model
        else:
            self.model_selector = None
            self.raw_model = model
            logger.warning("InferencePipeline initialized with a raw model.")
        
    @classmethod
    def load(cls, path: str, **kwargs):
        """Load a saved model and create an inference pipeline."""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return cls(model, **kwargs)

    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing, feature engineering, and calculate ACTUAL labels.
        """
        required_cols = ['Date', 'SecuritiesCode', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Handle Target column
        has_actual_target = 'Target' in df.columns and not (df['Target'] == 0).all()
        
        df_processed = df.copy()
        df_processed['Date'] = pd.to_datetime(df_processed['Date'])
        
        # 1. Compute VWAP & Features
        df_processed = compute_vwap(df_processed)
        df_processed = compute_feature_eng(df_processed, self.fast_period, self.slow_period)
        
        # 2. Fix Rank (Actual): Calculate actual rank if Target exists, else dummy
        if has_actual_target:
            # We must be careful to handle the rank the SAME way as training
            df_processed['Rank'] = df_processed.groupby("Date")["Target"].rank(
                ascending=False, method="first"
            ).astype(int) - 1
        else:
            if 'Rank' not in df_processed.columns:
                df_processed['Rank'] = 0
            
        return df_processed
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions and return a clean, sorted evaluation DataFrame.
        """
        # Preprocess features
        df_processed = self.preprocess_features(df)
        
        # Make predictions
        if self.model_selector is not None:
            predictions = self.model_selector.predict(df_processed)
        else:
            exclude = ['Date', 'Rank', 'Target']
            X_feat = df_processed.drop(columns=exclude, errors='ignore')
            predictions = self.raw_model.predict(X_feat)
        
        # 3. Combine results
        df_result = df_processed.copy()
        df_result['Score'] = predictions
        
        # 4. Create Predicted Rank (renamed for clarity)
        df_result['PredictedRank'] = df_result.groupby('Date')['Score'].rank(
            method='first', ascending=False
        ).astype(int) - 1
        
        # 5. Sort according to request: Date (asc), Target (desc), Rank (asc)
        # Higher target usually means Rank 0.
        df_result = df_result.sort_values(
            by=['Date', 'Target', 'Rank'], 
            ascending=[True, False, True]
        ).reset_index(drop=True)
        
        # Reorder columns to match training validation output style
        cols = ['Date', 'SecuritiesCode', 'Target', 'Rank', 'Score', 'PredictedRank']
        # Add any extra technical columns at the end if they exist
        existing_cols = [c for c in cols if c in df_result.columns]
        other_cols = [c for c in df_result.columns if c not in existing_cols]
        
        return df_result[existing_cols + other_cols]
    
    def predict_for_kaggle(self, prices_df: pd.DataFrame, 
                          sample_prediction: pd.DataFrame) -> pd.DataFrame:
        """
        Special format for Kaggle API submissions.
        """
        try:
            df_processed = self.preprocess_features(prices_df)
            if self.model_selector is not None:
                predictions = self.model_selector.predict(df_processed)
            else:
                exclude = ['Date', 'Rank', 'Target']
                X_feat = df_processed.drop(columns=exclude, errors='ignore')
                predictions = self.raw_model.predict(X_feat)
            
            result_df = sample_prediction.copy()
            pred_mapping = dict(zip(df_processed['SecuritiesCode'], predictions))
            result_df['Score'] = result_df['SecuritiesCode'].map(pred_mapping)
            result_df['Score'] = result_df['Score'].fillna(result_df['Score'].min() - 1)
            
            # Use 'Rank' for the final submisson column name as required by Kaggle
            result_df['Rank'] = result_df['Score'].rank(method='first', ascending=False).astype(int) - 1
            return result_df[['Rank']]
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            result_df = sample_prediction.copy()
            result_df['Rank'] = np.arange(len(result_df))
            return result_df[['Rank']]

    def _validate_ranks(self, ranks: pd.Series):
        """Validate ranking constraints."""
        n_stocks = len(ranks)
        if (ranks < 0).any() or (ranks >= n_stocks).any():
            raise ValueError(f"Ranks out of bounds [0, {n_stocks}-1]")
        if ranks.duplicated().any():
            raise ValueError("Duplicate ranks found")
