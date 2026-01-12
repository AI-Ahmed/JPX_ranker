import multiprocessing as mp
from typing import List, Optional, Tuple, Any

import pandas as pd
import numpy as np

import lightgbm as lgb
import catboost as cb

from loguru import logger

from pandarallel import pandarallel as ppl
from .utils import _calc_spread_return_per_day


ppl.initialize(progress_bar=False, nb_workers=mp.cpu_count() - 1)


class DataProcessor:
    """
    Unified data processor for LightGBM, XGBoost, and CatBoost ranking models.
    
    This class handles data preparation and conversion to framework-specific formats,
    including proper grouping for ranking tasks and categorical feature handling.
    
    Parameters
    ----------
    group_col : str, optional
        Column name to use for grouping in ranking tasks (default: 'Date').
    cat_features : list, optional
        List of categorical feature column names (default: ['SecuritiesCode']).
    target_col : str, optional
        Target column name (default: 'Rank').
    exclude_features : list, optional
        Features to exclude to prevent data leakage (default: ['Target']).
    
    Notes
    -----
    Each framework requires different data structures:
    - LightGBM: Dataset with group parameter, ranks must be 0 to N-1 per group
    - XGBoost: Separate X, y with qid for grouping
    - CatBoost: Pool with group_id parameter, categorical features as int/string
    
    The 'Target' column (future returns) is excluded by default to prevent data leakage.
    At prediction time, we won't have access to future returns, so they cannot be features.    
    """

    def __init__(self, 
                 group_col: str = 'Date', 
                 cat_features: List[str] | None = None,
                 target_col: str = 'Rank',
                 exclude_features: List[str] | None = None):
        self.group_col = group_col
        # Only SecuritiesCode is categorical; 'side' is numeric (-1.0, 1.0) and shouldn't be treated as categorical
        self.cat_features = cat_features or ['SecuritiesCode']
        self.target_col = target_col
        # Exclude 'Target' to prevent data leakage - it contains future returns
        self.exclude_features = exclude_features or ['Target']
        # Store category levels and feature names
        self.categories_ = {}
        self.feature_names_ = None
        # Store category levels for each categorical feature
        self.categories_ = {}
    
    def _compute_groups(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute group counts for ranking.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with group column.
            
        Returns
        -------
        np.ndarray
            Array of group sizes for each unique group.
            
        Notes
        -----
        Groups are computed based on unique values in the group column.
        The returned array contains the count of samples in each group.
        """
        return df.groupby(self.group_col).size().values
    
    def _get_group_indices(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get group indices for each sample.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with group column.
            
        Returns
        -------
        np.ndarray
            Array mapping each sample to its group index.
        """
        group_mapping = {group: idx for idx, group in enumerate(df[self.group_col].unique())}
        return df[self.group_col].map(group_mapping).values
    
    def _transform_labels(self, y: pd.Series, max_label: int) -> pd.Series:
        """
        Transform ranks to relevance labels (higher = better).
        
        Ranking frameworks (LightGBM, XGBoost, CatBoost) expect higher label = higher relevance.
        Our original ranks have: Rank 0 = best, Rank N = worst
        This method transforms to: Label max_label = best, Label 0 = worst
        
        Parameters
        ----------
        y : pd.Series
            Original rank labels (0 = best, N = worst).
        max_label : int
            Maximum label value in the dataset.
            
        Returns
        -------
        pd.Series
            Transformed relevance labels (max_label = best, 0 = worst).
            
        Notes
        -----
        This transformation is critical for correct NDCG calculation across all frameworks.
        Without it, NDCG scores can be invalid (>1.0 for LightGBM, 0.0 for CatBoost).
        """
        return max_label - y
    
    def _normalize_labels(self, y: pd.Series, max_label: int, 
                         num_grades: int = 30) -> pd.Series:
        """
        Normalize ranks to bounded relevance grades for LightGBM.
        
        LightGBM's lambdarank objective has limitations with large label ranges.
        This method normalizes ranks to a bounded range (e.g., 0-30) to avoid
        the "Label X is not less than the number of label mappings (Y)" error.
        
        Parameters
        ----------
        y : pd.Series
            Original rank labels (0 = best, N = worst).
        max_label : int
            Maximum label value in the dataset.
        num_grades : int, optional
            Number of relevance grades to use (default: 30).
            
        Returns
        -------
        pd.Series
            Normalized relevance labels in range [0, num_grades], where
            higher values indicate better relevance.
            
        Notes
        -----
        This addresses the LightGBM limitation where it expects labels in a
        bounded range by default. The transformation preserves relative ordering
        while compressing to manageable grades.
        """
        # Transform from rank (0=best) to relevance (high=best)
        relevance = max_label - y
        # Normalize to [0, num_grades]
        normalized = (relevance / max_label * num_grades).round().astype(int)
        return normalized.clip(0, num_grades)
    
    def prepare_lgb_dataset(self, 
                            df: pd.DataFrame, 
                            label: Optional[pd.Series] = None,
                            reference: Optional[lgb.Dataset] = None,
                            max_label: Optional[int] = None) -> Tuple[lgb.Dataset, Optional[List[int]]]:
        """
        Prepare LightGBM Dataset.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input features DataFrame (should include group column).
        label : pd.Series, optional
            Target labels. If None, uses self.target_col from df.
        reference : lgb.Dataset, optional
            Reference dataset for validation.
        max_label : int, optional
            Maximum label value for transformation. If provided, transforms ranks
            to relevance scores (higher = better) using simple reversal.
            
        Returns
        -------
        Tuple[lgb.Dataset, Optional[List[int]]]
            LightGBM Dataset object with proper grouping and categorical features,
            and optional label_gain list to pass to LGBMRanker parameters.
            
        Notes
        -----
        Data must be sorted by group_col for LightGBM's group parameter.
        If max_label is provided, ranks are transformed to relevance labels using
        simple reversal (max_label - rank) to preserve full granularity, matching
        XGBoost and CatBoost behavior.
        
        The label_gain parameter is returned separately to be passed to the
        LGBMRanker constructor, not to lgb.Dataset (which doesn't accept it).
        """
        # Sort by group column and SecuritiesCode - guarantees deterministic order
        df = df.sort_values(by=[self.group_col, 'SecuritiesCode']).reset_index(drop=True)
        
        # Keep group column for computing groups, but don't use as feature
        group_counts = self._compute_groups(df)
        
        # Drop group column, target, and excluded features (like 'Target' to prevent leakage)
        cols_to_drop = [self.group_col, self.target_col] + self.exclude_features
        X = df.drop(columns=cols_to_drop, errors='ignore')
        
        # Capture feature names if this is the first time (training)
        if self.feature_names_ is None:
            self.feature_names_ = X.columns.tolist()
        else:
            # Enforce exact feature set and order from training
            X = X[self.feature_names_]
            
        y = label if label is not None else df[self.target_col]
        
        # Transform labels to relevance scores (higher = better) - preserves full granularity
        if max_label is not None:
            y = self._transform_labels(y, max_label)
        
        # Convert categorical features to category dtype for LightGBM
        for cat_col in self.cat_features:
            if cat_col in X.columns:
                if cat_col not in self.categories_:
                    # Training mode - let pandas discover categories
                    X[cat_col] = X[cat_col].astype('category')
                    self.categories_[cat_col] = X[cat_col].cat.categories
                else:
                    # Inference/Validation mode - use stored categories
                    X[cat_col] = pd.Categorical(X[cat_col], categories=self.categories_[cat_col])
        
        # Create label_gain for LightGBM to handle large label ranges
        # LightGBM's lambdarank has a 31-label limit by default, but we can override with label_gain
        # Use linear gains: label i gets gain i (higher label = higher gain)
        # This must be passed to LGBMRanker constructor, NOT to Dataset
        label_gain = None
        if max_label is not None and max_label > 30:
            # Create gains from 0 to max_label (linear gains)
            label_gain = list(range(max_label + 1))
        
        dataset = lgb.Dataset(
            X,
            label=y,
            group=group_counts,
            categorical_feature=self.cat_features,
            reference=reference,
            free_raw_data=False
        )
        
        return dataset, label_gain
    
    def prepare_xgb_data(self, 
                         df: pd.DataFrame, 
                         label: Optional[pd.Series] = None,
                         max_label: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """
        Prepare data for XGBoost Ranker (returns X, y, qid separately).
        
        Parameters
        ----------
        df : pd.DataFrame
            Input features DataFrame (should include group column).
        label : pd.Series, optional
            Target labels. If None, uses self.target_col from df.
        max_label : int, optional
            Maximum label value for transformation. If provided, transforms ranks
            to relevance scores (higher = better).
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.Series, np.ndarray]
            X (features), y (labels), qid (group identifiers) for XGBRanker.fit()
            
        Notes
        -----
        XGBRanker.fit() expects separate X and y arguments, not DMatrix.
        The qid (query id) parameter maps each sample to its group.
        Data is sorted by group for consistency.
        If max_label is provided, ranks are transformed to relevance labels.
        """
        # Sort by group column and SecuritiesCode for deterministic order
        df = df.sort_values(by=[self.group_col, 'SecuritiesCode']).reset_index(drop=True)
        
        # Get group identifiers (qid) before dropping group column
        qid = self._get_group_indices(df)
        
        # Drop group column, target, and excluded features (like 'Target' to prevent leakage)
        cols_to_drop = [self.group_col, self.target_col] + self.exclude_features
        X = df.drop(columns=cols_to_drop, errors='ignore')
        y = label if label is not None else df[self.target_col]
        
        # Transform labels if max_label is provided
        if max_label is not None:
            y = self._transform_labels(y, max_label)
        
        # XGBoost requires enable_categorical=True for category dtype
        # For simplicity and compatibility, convert categorical features to int
        for cat_col in self.cat_features:
            if cat_col in X.columns:
                # Convert to int if numeric (like SecuritiesCode)
                if pd.api.types.is_numeric_dtype(X[cat_col]):
                    X[cat_col] = X[cat_col].astype(int)
                else:
                    # For string categories, use label encoding
                    X[cat_col] = pd.Categorical(X[cat_col]).codes
        
        return X, y, qid
    
    def prepare_catboost_pool(self, 
                              df: pd.DataFrame, 
                              label: Optional[pd.Series] = None,
                              max_label: Optional[int] = None) -> cb.Pool:
        """
        Prepare CatBoost Pool.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input features DataFrame (should include group column).
        label : pd.Series, optional
            Target labels. If None, uses self.target_col from df.
        max_label : int, optional
            Maximum label value for transformation. If provided, transforms ranks
            to relevance scores (higher = better).
            
        Returns
        -------
        cb.Pool
            CatBoost Pool object with proper grouping and categorical features.
            
        Notes
        -----
        CatBoost uses group_id parameter which should be the group column itself,
        not the group counts. It automatically handles grouping internally.
        Categorical features must be integers or strings - floats are converted to strings.
        Data is sorted by group for consistency.
        If max_label is provided, ranks are transformed to relevance labels.
        """
        # Sort by group column and SecuritiesCode for deterministic order
        df = df.sort_values(by=[self.group_col, 'SecuritiesCode']).reset_index(drop=True)
        
        # Drop target and excluded features (like 'Target' to prevent leakage), but keep group column for group_id
        cols_to_drop = [self.target_col] + self.exclude_features
        X = df.drop(columns=cols_to_drop, errors='ignore')
        y = label if label is not None else df[self.target_col]
        
        # Transform labels if max_label is provided
        # CatBoost YetiRank: If labels are in [0,1], it uses PFound; otherwise NDCG
        # We need labels > 1 to ensure NDCG is used, so add 2 to transformed labels
        if max_label is not None:
            y = self._transform_labels(y, max_label) + 2  # Shift to ensure values > 1
        
        # Prepare features (drop group column)
        X_features = X.drop(columns=[self.group_col]).copy()
        
        # Convert categorical features to string type (CatBoost requirement)
        # CatBoost expects categorical features to be integers or strings, not floats
        cat_feature_indices = []
        for cat_col in self.cat_features:
            if cat_col in X_features.columns:
                # If the column is numeric, convert to int first (if possible), then to string
                if pd.api.types.is_numeric_dtype(X_features[cat_col]):
                    # Ensure it's an integer-compatible type before converting
                    try:
                        X_features[cat_col] = X_features[cat_col].astype(int).astype(str)
                    except (ValueError, OverflowError):
                        # If conversion fails, skip this as categorical feature
                        logger.warning(f"Skipping {cat_col} as categorical - contains non-integer values")
                        continue
                else:
                    X_features[cat_col] = X_features[cat_col].astype(str)
                
                # Add to categorical feature indices
                cat_feature_indices.append(X_features.columns.get_loc(cat_col))
        
        # Convert group_id to string (CatBoost requires string or int, not datetime)
        group_id_values = X[self.group_col].astype(str).values
        
        pool = cb.Pool(
            data=X_features,
            label=y,
            group_id=group_id_values,
            cat_features=cat_feature_indices
        )
        
        return pool
    
    def prepare_data(self, 
                     df: pd.DataFrame, 
                     framework: str,
                     label: Optional[pd.Series] = None,
                     reference: Optional[Any] = None) -> Any:
        """
        Prepare data for specified framework.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input features DataFrame.
        framework : str
            Framework name: 'lightgbm', 'xgboost', or 'catboost'.
        label : pd.Series, optional
            Target labels.
        reference : Any, optional
            Reference dataset (only used for LightGBM).
            
        Returns
        -------
        Any
            Framework-specific data object:
            - LightGBM: lgb.Dataset
            - XGBoost: Tuple[pd.DataFrame, pd.Series, np.ndarray] (X, y, qid)
            - CatBoost: cb.Pool
            
        Raises
        ------
        ValueError
            If framework is not recognized.
        """
        framework = framework.lower()
        
        if framework in ['lightgbm', 'lgb', 'lgbmranker']:
            return self.prepare_lgb_dataset(df, label, reference)
        elif framework in ['xgboost', 'xgb', 'xgbranker']:
            return self.prepare_xgb_data(df, label)
        elif framework in ['catboost', 'cb', 'catboostranker']:
            return self.prepare_catboost_pool(df, label)
        else:
            raise ValueError(f"Unknown framework: {framework}. Use 'lightgbm', 'xgboost', or 'catboost'.")
