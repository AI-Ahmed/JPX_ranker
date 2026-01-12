import pickle
from typing import Tuple, Optional

import pandas as pd
import numpy as np

from .model_selection import ModelSelector
from .utils import calc_spread_return_sharpe


class TrainingPipeline:
    """
    Professional training pipeline for ranking models.
    
    This class orchestrates the entire training process including data splitting,
    model training, evaluation, and persistence.
    
    Parameters
    ----------
    test_size : float, optional
        Proportion of data to use for validation (default: 0.2).
    n_trials : int, optional
        Number of Optuna trials per model (default: 200).
    n_folds : int, optional
        Number of folds for CombinatorialPurgedCV (default: 10).
    n_test_folds : int, optional
        Number of test folds for each split (default: 2).
    purged_size : int, optional
        Number of observations to exclude around test sets to prevent leakage (default: 2; 
        because we do have lookahead of 2 days).
    embargo_size : int, optional
        Number of observations to exclude after test sets due to serial correlation (default: 21; 
        because we do have features with 21 days lookback [e.g., `autocorr-21`]).
    seed : int, optional
        Random seed for reproducibility (default: 42).
        
    Attributes
    ----------
    model_selector_ : ModelSelector
        Fitted ModelSelector instance.
    train_data_ : pd.DataFrame
        Training data after split.
    val_data_ : pd.DataFrame
        Validation data after split.
    train_score_ : float
        Training Sharpe ratio.
    val_score_ : float
        Validation Sharpe ratio.
    """
    
    def __init__(self, test_size=0.2, n_trials=200, n_folds=10, 
                 n_test_folds=2, purged_size=2, embargo_size=21,
                 seed=42, device='cpu', min_trials=5):
        self.test_size = test_size
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.n_test_folds = n_test_folds
        self.purged_size = purged_size
        self.embargo_size = embargo_size
        self.seed = seed
        self.device = device
        self.min_trials = min_trials
        
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data by date for time series validation.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with 'Date' column.
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Training and validation dataframes.
            
        Notes
        -----
        Split is performed chronologically to maintain time series structure.
        The last `test_size` portion of dates is used for validation.
        """
        unique_dates = sorted(df['Date'].unique())
        split_idx = int(len(unique_dates) * (1 - self.test_size))
        split_date = unique_dates[split_idx]
        
        train_df = df[df['Date'] < split_date].copy()
        val_df = df[df['Date'] >= split_date].copy()
        
        print(f"Train samples: {len(train_df):,} | Val samples: {len(val_df):,}")
        print(f"Train dates: {train_df['Date'].min()} to {train_df['Date'].max()}")
        print(f"Val dates: {val_df['Date'].min()} to {val_df['Date'].max()}")
        
        return train_df, val_df
    
    def evaluate_predictions(self, df: pd.DataFrame, predictions: np.ndarray,
                            name: str = "Evaluation") -> float:
        """
        Evaluate predictions using Sharpe ratio.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with actual data and 'Date', 'Target', 'Rank' columns.
        predictions : np.ndarray
            Predicted ranking scores.
        name : str, optional
            Name for logging (default: "Evaluation").
            
        Returns
        -------
        float
            Sharpe ratio of the predictions.
        """
        df_eval = df.copy()
        
        # Use predicted scores to rank stocks within each date
        df_eval['Score'] = predictions
        df_eval['PredictedRank'] = df_eval.groupby('Date')['Score'].rank(method='first', ascending=False) - 1
        df_eval['PredictedRank'] = df_eval['PredictedRank'].astype(int)
        
        # Calculate Sharpe ratio - select only needed columns and rename to avoid duplicate 'Rank'
        # We need: Date, Target, and the predicted rank (renamed to 'Rank' for the function)
        sharpe_df = df_eval[['Date', 'Target', 'PredictedRank']].copy()
        sharpe_df.rename(columns={'PredictedRank': 'Rank'}, inplace=True)
        sharpe = calc_spread_return_sharpe(sharpe_df)
        
        print(f"{name} Sharpe Ratio: {sharpe:.6f}")
        return sharpe
    
    def fit(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Fit the training pipeline.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input training data.
        save_path : str, optional
            Path to save the trained model (default: None).
            
        Returns
        -------
        self
            Fitted TrainingPipeline instance.
        """
        print("="*80)
        print("TRAINING PIPELINE START")
        print("="*80)
        
        # Split data
        print("\n[1/4] Splitting data...")
        self.train_data_, self.val_data_ = self.split_data(df)
        
        # Train model
        print("\n[2/4] Training models...")
        self.model_selector_ = ModelSelector(
            n_trials=self.n_trials,
            n_folds=self.n_folds,
            n_test_folds=self.n_test_folds,
            purged_size=self.purged_size,
            embargo_size=self.embargo_size,
            seed=self.seed,
            device=self.device,
            min_trials=self.min_trials
        )
        
        self.model_selector_.fit(
            X=self.train_data_,
            X_val=self.val_data_
        )
        
        # Evaluate on training set
        print("\n[3/4] Evaluating on training set...")
        train_predictions = self.model_selector_.predict(self.train_data_)
        self.train_score_ = self.evaluate_predictions(
            self.train_data_, train_predictions, "Training"
        )
        
        # Evaluate on validation set
        print("\n[4/4] Evaluating on validation set...")
        val_predictions = self.model_selector_.predict(self.val_data_)
        self.val_score_ = self.evaluate_predictions(
            self.val_data_, val_predictions, "Validation"
        )
        
        # Save model
        if save_path:
            print(f"\nSaving model to {save_path}...")

            with open(save_path, 'wb') as f:
                pickle.dump(self.model_selector_, f)
            print("Model saved successfully!")
        
        print("\n" + "="*80)
        print("TRAINING PIPELINE COMPLETE")
        print("="*80)
        print(f"Best Model: {self.model_selector_.best_model_name_}")
        print(f"Best NDCG@100: {self.model_selector_.best_score_:.6f}")
        print(f"Training Sharpe: {self.train_score_:.6f}")
        print(f"Validation Sharpe (Outer Holdout): {self.val_score_:.6f}")
        
        # Display statistical validation summary
        if hasattr(self.model_selector_, 'adjusted_metrics_'):
            metrics = self.model_selector_.adjusted_metrics_
            print(f"\nStatistical Validation Metrics:")
            print(f"  Deflated Sharpe Ratio: {metrics['deflated_sharpe_ratio']:.4f}")
            print(f"  Observed FDR: {metrics['observed_fdr']:.4f}")
            print(f"  Statistically Significant: {metrics['significant']}")
        
        print("="*80)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using the trained model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features for prediction.
            
        Returns
        -------
        np.ndarray
            Predicted ranking scores.
        """
        if not hasattr(self, 'model_selector_'):
            raise ValueError("Pipeline not fitted yet. Call fit() first.")
        
        return self.model_selector_.predict(X)
