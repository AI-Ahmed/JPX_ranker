import traceback

import numpy as np
import pandas as pd
import scipy.stats

from sklearn.base import BaseEstimator

import optuna
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from skfolio.model_selection import CombinatorialPurgedCV

from loguru import logger

from .data_prep import DataProcessor
from .utils import (
    calc_spread_return_sharpe, 
    calc_spread_return_sharpe_scorer,
    _calc_spread_return_per_day
)
from .statistical_validation import (
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    expected_maximum_sharpe_ratio,
    oFDR
)


class MinTRLCallback:
    """
    Optuna callback for early stopping based on Minimum Track Record Length.
    
    Stops optimization when we have enough trials to be statistically confident
    that the best Sharpe ratio is significantly better than the null hypothesis.
    
    Parameters
    ----------
    selector : ModelSelector
        Reference to the ModelSelector instance for accessing validation data
    SR0 : float, optional
        Null hypothesis Sharpe ratio (default: 0 = no skill)
    alpha : float, optional
        Significance level (default: 0.05)
    significance_threshold : float, optional
        PSR threshold for early stopping (default: 0.95)
    min_trials : int, optional
        Minimum trials before checking (default: 10)
        
    References
    ----------
    .. [1] Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio: 
            Correcting for Selection Bias, Backtest Overfitting and Non-Normality".
            Journal of Portfolio Management, 40(5), 94-107. 
   
    Examples
    --------
    >>> callback = MinTRLCallback(selector, SR0=0, alpha=0.05, significance_threshold=0.95)
    >>> study.optimize(objective, n_trials=200, callbacks=[callback])
    """
    
    def __init__(self, selector, SR0=0, alpha=0.05, significance_threshold=0.95, min_trials=5):
        self.selector = selector
        self.SR0 = SR0
        self.alpha = alpha
        self.significance_threshold = significance_threshold
        self.min_trials = min_trials
    
    def __call__(self, study, trial):
        """
        Check if we should stop optimization based on MinTRL.
        
        Parameters
        ----------
        study : optuna.Study
            Optuna study object
        trial : optuna.Trial
            Current trial object
        """        
        # The objective function has already appended the sharpe to the selector's global list
        # We check the global trial count across all models to correctly account for selection bias (K)
        if len(self.selector.trial_sharpes_) < self.min_trials:
            # Log every few trials to show the callback is working
            logger.debug(f"MinTRL: Collecting trials ({len(self.selector.trial_sharpes_)}/{self.min_trials} needed before stopping check)")
            return  # Need minimum samples
        
        # Get number of trials (K) and validation days (T)
        K = len(self.selector.trial_sharpes_)
        
        # Get T (number of validation days) - computed on first trial
        if self.selector.validation_T_ is None or self.selector.validation_T_ <= 0:
            return  # T not yet computed, wait for more trials
        T = self.selector.validation_T_
        
        # Compute gamma3/gamma4 from actual validation returns (not from Sharpe ratios!)
        if len(self.selector.validation_returns_) > 0:
            # Use most recent validation returns
            daily_returns = self.selector.validation_returns_[-1]
            if len(daily_returns) > 0:
                gamma3 = scipy.stats.skew(daily_returns)
                gamma4 = scipy.stats.kurtosis(daily_returns, fisher=False)
            else:
                # Fallback to normal distribution assumptions
                gamma3, gamma4 = 0.0, 3.0
        else:
            # Fallback to normal distribution assumptions
            gamma3, gamma4 = 0.0, 3.0
        
        # For multi-objective, find best Sharpe across all trials/models
        best_sr = max(self.selector.trial_sharpes_)
        
        # Validation checks
        if T <= 0 or K <= 0:
            return
        
        # Ensure T and K are not confused (they should be different unless T=1)
        if T == K and T > 1:
            logger.warning(f"T ({T}) equals K ({K}), which suggests parameter confusion")
            return
        
        # Check MinTRL
        min_trl = minimum_track_record_length(
            best_sr, self.SR0, gamma3=gamma3, gamma4=gamma4, alpha=self.alpha
        )
        
        # Log status every few trials
        psr = probabilistic_sharpe_ratio(
            best_sr, self.SR0, T=T, gamma3=gamma3, gamma4=gamma4, K=K
        )
        
        # Enforce per-model minimum exploration to prevent architecture starvation
        # Every model architecture deserves a chance to be tuned locally before 
        # the global "Strategic Curse" detector shuts it down.
        trials_this_model = len(study.trials)
        min_per_model = 5  # Minimum exploration per architecture
        
        if K % 5 == 0:
            logger.debug(f"MinTRL Status (K_global={K}, K_model={trials_this_model}): "
                        f"Best SR={best_sr:.4f}, PSR={psr:.3f}, T={T}/{min_trl:.0f} needed")
        
        if T >= min_trl and psr > self.significance_threshold:
            if trials_this_model < min_per_model:
                logger.debug(f"MinTRL: Significance reached but model needs more exploration "
                            f"({trials_this_model}/{min_per_model})...")
                return
            
            logger.info(f"\n✓ Early stopping: {study.study_name} reached MinTRL significance. "
                        f"(T={T}, PSR={psr:.3f}, K_global={K}, K_model={trials_this_model})")
            study.stop()  # Statistically significant - stop early


class ModelSelector(BaseEstimator):
    """
    Model selector for ranking tasks using LightGBM, XGBoost, and CatBoost.
    
    This class trains multiple ranking models with Optuna hyperparameter optimization
    and selects the best performing model based on NDCG@100 metric.
    
    Uses CombinatorialPurgedCV for robust time series cross-validation that prevents
    data leakage through purging and embargoing techniques [1].
    
    Parameters
    ----------
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
    eval_metric : str, optional
        Evaluation metric for ranking (default: 'ndcg').
    eval_at : list, optional
        List of k values for NDCG@k evaluation (default: [10, 50, 100]).
        
    Attributes
    ----------
    best_model_ : object
        Best trained model instance.
    best_params_ : dict
        Best hyperparameters found.
    best_score_ : float
        Best NDCG@100 score achieved.
    best_model_name_ : str
        Name of the best model.
    data_processor_ : DataProcessor
        Data processor instance for preparing datasets.
    model_results_ : dict
        Dictionary storing results for all models.
    cv : CombinatorialPurgedCV
        Cross-validation splitter instance.
        
    References
    ----------
    .. [1] López de Prado, M. (2018). "Advances in Financial Machine Learning". 
           Wiley. https://skfolio.org/generated/skfolio.model_selection.CombinatorialPurgedCV.html
           
    Notes
    -----
    CombinatorialPurgedCV offers significant advantages over standard TimeSeriesSplit:
    
    - **Purging**: Removes overlapping labels from training set that occurred during test period
    - **Embargoing**: Removes observations immediately following test set to handle serial correlation
    - **Combinatorial paths**: Tests multiple train/test combinations for robust validation
    - **Financial ML**: Specifically designed for financial time series with overlapping labels
    """
    
    def __init__(self, n_trials=200, n_folds=10, n_test_folds=2, 
                 purged_size=2, embargo_size=21, seed=42, 
                 eval_metric='ndcg', eval_at=None,
                 device='cpu', min_trials=5):
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.n_test_folds = n_test_folds
        self.purged_size = purged_size
        self.embargo_size = embargo_size
        self.seed = seed
        self.eval_metric = eval_metric
        self.eval_at = eval_at or [10, 50, 100]
        self.device = device
        self.min_trials = min_trials

        # Use CombinatorialPurgedCV for better time series validation
        # Ref: https://skfolio.org/generated/skfolio.model_selection.CombinatorialPurgedCV.html
        self.cv = CombinatorialPurgedCV(
            n_folds=self.n_folds,
            n_test_folds=self.n_test_folds,
            purged_size=self.purged_size,
            embargo_size=self.embargo_size
        )
        self.data_processor_ = DataProcessor()
        
        # Statistical validation attributes
        self.SR0 = 0.0  # Null hypothesis: no skill
        self.SR1 = 0.5  # Alternative: meaningful strategy
        self.p_H1 = 0.05  # Prior probability of good strategy
        self.trial_sharpes_ = []  # Track Sharpe ratios across trials
        self.trial_ndcgs_ = []  # Track NDCG scores across trials
        self.cv_scores_ = {}  # Per-fold scores
        self.n_paths_tested_ = 0  # Number of CV paths evaluated
        self.max_cv_paths = None  # Optional: limit CV paths for speed
        
        # Validation data for correct statistical validation
        self.validation_returns_ = []  # Store daily spread returns from validation
        self.validation_T_ = None  # Number of validation trading days (computed on first trial)

    def _compute_return_statistics(self, df, portfolio_size=200, toprank_weight_ratio=2):
        """
        Compute return distribution statistics (γ₃, γ₄) from daily spread returns.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'Date', 'Rank', and 'Target' columns
        portfolio_size : int, optional
            Number of equities to buy (long) and sell (short) (default: 200)
        toprank_weight_ratio : float, optional
            Weight ratio between top and bottom ranked stocks (default: 2)
            
        Returns
        -------
        dict
            Dictionary with gamma3 (skewness) and gamma4 (kurtosis)
            
        Notes
        -----
        These statistics are used in López de Prado's statistical validation framework
        to account for non-normal return distributions when computing Sharpe ratio variance.
        
        For a ranking strategy, we compute statistics from the **daily spread returns**
        (long-short portfolio returns for each date), NOT from individual security returns.
        This matches the actual returns that generate the Sharpe ratio being validated.
        
        The spread return strategy goes long on top-ranked securities and short on
        bottom-ranked securities with linearly decreasing weights.
        """        
        # Compute daily spread returns
        daily_spread_returns = df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
        
        gamma3 = scipy.stats.skew(daily_spread_returns)
        gamma4 = scipy.stats.kurtosis(daily_spread_returns, fisher=False)  # Non-excess kurtosis
        return {'gamma3': gamma3, 'gamma4': gamma4}

    def _compute_cv_score(self, model_class, params, df, max_label):
        """
        Compute mean NDCG and Sharpe across all CPCV folds.
        
        Parameters
        ----------
        model_class : str
            Model class name ('lgbm', 'xgb', 'catboost')
        params : dict
            Model hyperparameters
        df : pd.DataFrame
            Full training data
        max_label : int
            Maximum label for transformation
            
        Returns
        -------
        dict
            Dictionary with mean/std NDCG and Sharpe scores across folds
            
        Notes
        -----
        This method implements the core CPCV integration. It:
        1. Splits data using CombinatorialPurgedCV
        2. Trains model on each fold
        3. Evaluates NDCG and Sharpe on each validation fold
        4. Returns aggregated statistics
        """        
        ndcg_scores, sharpe_diffs, sharpe_scores = [], [], []
        all_daily_spread_returns = []  # Collect daily returns from all folds
        unique_dates = sorted(df['Date'].unique())
        # Convert to numpy array for proper fancy indexing
        unique_dates = np.array(unique_dates)
        X_indices = np.arange(len(unique_dates))
        
        fold_count = 0
        max_cv_paths = getattr(self, 'max_cv_paths', None)
        
        for train_idx, test_idx in self.cv.split(X_indices):
            if max_cv_paths and fold_count >= max_cv_paths:
                break
            
            # Convert indices to proper numpy integer arrays for indexing
            # CombinatorialPurgedCV returns test_idx as a list of arrays
            train_idx = np.asarray(train_idx, dtype=int).flatten()
            if isinstance(test_idx, list):
                # Concatenate multiple test fold indices
                test_idx = np.concatenate([np.asarray(idx, dtype=int).flatten() for idx in test_idx])
            else:
                test_idx = np.asarray(test_idx, dtype=int).flatten()
                
            train_dates = unique_dates[train_idx]
            test_dates = unique_dates[test_idx]
            
            train_fold = df[df['Date'].isin(train_dates)]
            val_fold = df[df['Date'].isin(test_dates)]
            
            (ndcg,
             sharpe_diff,
             sharpe_pred,
             daily_spread_returns) = self._evaluate_fold(
                model_class,
                params,
                train_fold,
                val_fold,
                max_label
            )
            ndcg_scores.append(ndcg)
            sharpe_diffs.append(sharpe_diff)
            sharpe_scores.append(sharpe_pred)
            all_daily_spread_returns.append(daily_spread_returns)
            fold_count += 1
            
            # Store validation_T_ on first fold
            if self.validation_T_ is None and len(daily_spread_returns) > 0:
                self.validation_T_ = len(daily_spread_returns)
        
        self.n_paths_tested_ = fold_count
        
        # Aggregate daily returns from all folds (concatenate)
        if all_daily_spread_returns:
            aggregated_returns = np.concatenate([returns.values if hasattr(returns, 'values') else returns 
                                                for returns in all_daily_spread_returns if len(returns) > 0])
        else:
            aggregated_returns = np.array([])
        
        return {
            'ndcg_mean': np.mean(ndcg_scores),
            'ndcg_std': np.std(ndcg_scores),
            'sharpe_diff_mean': np.mean(sharpe_diffs),
            'sharpe_diff_std': np.std(sharpe_diffs),
            'sharpe_mean': np.mean(sharpe_scores),
            'sharpe_std': np.std(sharpe_scores),
            'daily_spread_returns': aggregated_returns  # Return aggregated daily returns
        }
    
    def _evaluate_fold(self, model_class, params, train_fold, val_fold, max_label):
        """
        Evaluate a single CV fold.
        
        Parameters
        ----------
        model_class : str
            Model class name ('lgbm', 'xgb', 'catboost')
        params : dict
            Model hyperparameters
        train_fold : pd.DataFrame
            Training fold data
        val_fold : pd.DataFrame
            Validation fold data
        max_label : int
            Maximum label for transformation
            
        Returns
        -------
        tuple
            (ndcg_score, sharpe_ratio, daily_returns) for this fold
            
        Notes
        -----
        This method trains a model on a single fold and evaluates both
        NDCG (for model selection) and Sharpe ratio (for statistical validation).
        Also returns daily spread returns for computing gamma3/gamma4.
        """        
        # Sort folds deterministically to ensure alignment between dataframe and model predictions
        train_fold = train_fold.sort_values(by=['Date', 'SecuritiesCode']).reset_index(drop=True)
        val_fold = val_fold.sort_values(by=['Date', 'SecuritiesCode']).reset_index(drop=True)
        
        # Train model on this fold
        # Store processed validation features for prediction
        X_val_for_pred = None
        
        # LEAKAGE CHECK: Ensure Target is not in train/val data features
        target_col = self.data_processor_.target_col
        # We can't easily check 'Target' column here as it's not yet processed, 
        # but we can rely on verifying X_val_for_pred later or checking DataProcessor config.
        if target_col not in self.data_processor_.exclude_features and 'Target' not in self.data_processor_.exclude_features:
            logger.warning(f"POTENTIAL LEAKAGE: '{target_col}' or 'Target' not in exclude_features!")

        
        if model_class == 'lgbm':
            train_data, label_gain = self.data_processor_.prepare_lgb_dataset(train_fold, max_label=max_label)
            val_data, _ = self.data_processor_.prepare_lgb_dataset(val_fold, reference=train_data, max_label=max_label)
            
            X_train, y_train = train_data.data, train_data.label
            X_val, y_val = val_data.data, val_data.label
            X_val_for_pred = X_val  # Store for prediction later
            group_train, group_val = train_data.get_group(), val_data.get_group()
            
            cat_features = [col for col in self.data_processor_.cat_features if col in X_train.columns]
            
            model = lgb.LGBMRanker(
                objective='lambdarank',
                metric=self.eval_metric,
                device='cpu', # GPU is not supported for large bin sizes
                n_jobs=-1,
                random_state=self.seed,
                verbose=-1,
                label_gain=label_gain,  # Pass label_gain to model, not Dataset
                **params
            )
            model.fit(
                X_train, y_train,
                group=group_train,
                eval_set=[(X_val, y_val)],
                eval_group=[group_val],
                eval_metric=self.eval_metric,
                eval_at=self.eval_at,
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
                categorical_feature=cat_features
            )
            ndcg_score = model.best_score_['valid_0'][f'{self.eval_metric}@{self.eval_at[-1]}']
            
        elif model_class == 'xgb':
            X_train, y_train, qid_train = self.data_processor_.prepare_xgb_data(train_fold, max_label=max_label)
            X_val, y_val, qid_val = self.data_processor_.prepare_xgb_data(val_fold, max_label=max_label)
            X_val_for_pred = X_val  # Store for prediction later
            
            model = xgb.XGBRanker(
                objective='rank:ndcg',
                tree_method='hist' if self.device.lower() == 'gpu' else 'auto',
                n_jobs=-1,
                random_state=self.seed,
                ndcg_exp_gain=False,
                early_stopping_rounds=30,
                **params
            )
            model.fit(
                X_train, y_train,
                qid=qid_train,
                eval_set=[(X_val, y_val)],
                eval_qid=[qid_val],
                verbose=False
            )
            # Get best validation score (with early stopping)
            # XGBoost stores best_score as a float after early stopping
            ndcg_score = model.best_score if hasattr(model, 'best_score') else 0.0
                
        elif model_class == 'catboost':
            train_pool = self.data_processor_.prepare_catboost_pool(train_fold, max_label=max_label)
            val_pool = self.data_processor_.prepare_catboost_pool(val_fold, max_label=max_label)
            # For CatBoost, we'll use the Pool directly for prediction
            # Store the pool itself since get_features() doesn't work with categorical features
            X_val_for_pred = val_pool
            
            ndcg_metric = f'NDCG:top={self.eval_at[-1]}'
            model = cb.CatBoostRanker(
                loss_function='YetiRank',
                eval_metric=ndcg_metric,
                task_type=self.device.upper(),
                random_seed=self.seed,
                verbose=False,
                **params
            )
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=30, verbose=False)
            
            # Get best validation score (similar to LightGBM's best_score_)
            # CatBoost can use different keys: 'validation', 'validation_0', or 'learn'
            best_scores = model.get_best_score()
            ndcg_score = 0.0
            
            if best_scores:
                # Try different possible validation keys
                for val_key in ['validation', 'validation_0', 'learn']:
                    if val_key in best_scores:
                        val_scores = best_scores[val_key]
                        # Try exact metric key first
                        if ndcg_metric in val_scores:
                            ndcg_score = val_scores[ndcg_metric]
                            break
                        else:
                            # Fallback: try to find any NDCG metric
                            for key in val_scores.keys():
                                if 'NDCG' in key.upper():
                                    ndcg_score = val_scores[key]
                                    break
                            if ndcg_score > 0:
                                break
            
            # If still 0, try evals_result_ as final fallback
            if ndcg_score == 0.0 and hasattr(model, 'evals_result_'):
                evals = model.evals_result_
                for eval_key in ['validation', 'validation_0', 'learn']:
                    if eval_key in evals:
                        val_evals = evals[eval_key]
                        if ndcg_metric in val_evals:
                            ndcg_score = max(val_evals[ndcg_metric])
                            break
                        else:
                            # Try any NDCG key
                            for key in val_evals.keys():
                                if 'NDCG' in key.upper():
                                    ndcg_score = max(val_evals[key])
                                    break
                        if ndcg_score > 0:
                            break
        else:
            raise ValueError(f"Unknown model class: {model_class}")
        
        # Compute Sharpe ratio for this fold
        # Use the processed validation features that were used during training
        if X_val_for_pred is not None:
            val_pred = model.predict(X_val_for_pred)
        else:
            # Fallback: prepare data the same way (shouldn't reach here)
            exclude_cols = ['Date', 'Rank'] + self.data_processor_.exclude_features
            val_pred = model.predict(val_fold.drop(columns=exclude_cols, errors='ignore'))

        val_fold_copy = val_fold.copy()
        val_fold_copy['Score'] = val_pred
        val_fold_copy['PredictedRank'] = val_fold_copy.groupby('Date')['Score'].rank(
            method='first', ascending=False) - 1
        val_fold_copy['PredictedRank'] = val_fold_copy['PredictedRank'].astype(int)
        
        predicted_df = val_fold_copy[['Date', 'Target', 'PredictedRank']].copy()
        predicted_df.rename(columns={'PredictedRank': 'Rank'}, inplace=True)
        
        # Actual DataFrame (Rank col expected in val_fold)
        actual_df = val_fold[['Date', 'Target', 'Rank']].copy()
        
        try:
            # Compute Sharpe Ratio Difference (Objective: Minimize)
            sharpe_diff = calc_spread_return_sharpe_scorer(actual_df, predicted_df)
            
            # Compute Predicted Sharpe Ratio (For Tracking/MinTRL/Validation)
            predicted_sharpe = calc_spread_return_sharpe(predicted_df)
            
            # Also compute daily spread returns for statistical validation (from predicted)
            daily_spread_returns = predicted_df.groupby('Date').apply(_calc_spread_return_per_day, 200, 2)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(f"Could not compute Sharpe metrics for fold: {e}")
            sharpe_diff = float('inf')
            predicted_sharpe = 0.0
            daily_spread_returns = np.array([])
        
        return ndcg_score, sharpe_diff, predicted_sharpe, daily_spread_returns
    
    def _compute_selection_bias_adjusted_metrics(self, trial_sharpes, best_sharpe, T_validation, validation_returns):
        """
        Compute Deflated Sharpe Ratio and oFDR after optimization.
        
        Adjusts for selection bias from multiple testing (K Optuna trials).
        
        Parameters
        ----------
        trial_sharpes : list
            List of Sharpe ratios from all trials
        best_sharpe : float
            Best Sharpe ratio achieved
        T_validation : int
            Number of validation observations (trading days)
        validation_returns : np.ndarray
            Daily spread returns from validation data
            
        Returns
        -------
        dict
            Dictionary with adjusted metrics:
            - raw_sharpe: Unadjusted Sharpe ratio
            - expected_max_sr: Expected inflation from K trials
            - deflated_sharpe_ratio: PSR adjusted for selection bias
            - observed_fdr: Probability of false discovery
            - gamma3, gamma4: Return distribution statistics
            - n_trials: Number of trials
            - significant: Whether result is statistically significant
            
        References
        ----------
        López de Prado, M. (2018). Advances in Financial Machine Learning, Ch. 7
        López de Prado, M., Lipton, A., & Zoonekynd, V. (2025). "Sharpe Ratio Inference: 
        A New Standard for Decision-Making and Reporting." Available at SSRN: 
        https://ssrn.com/abstract=5520741
        """
        K = len(trial_sharpes)
        sharpes = np.array(trial_sharpes)
        
        # Compute gamma3/gamma4 from actual validation returns (not from Sharpe ratios!)
        if len(validation_returns) > 0:
            gamma3 = scipy.stats.skew(validation_returns)
            gamma4 = scipy.stats.kurtosis(validation_returns, fisher=False)
        else:
            # Fallback to normal distribution assumptions
            gamma3, gamma4 = 0.0, 3.0
        
        # Variance of Sharpe ratios for expected maximum SR calculation
        # This is correct - we need variance of SRs across trials for the haircut
        variance = np.var(sharpes)
        
        # Expected maximum SR (haircut for selecting best of K)
        E_max_SR = expected_maximum_sharpe_ratio(K, variance, SR0=self.SR0)
        
        # Deflated Sharpe Ratio: PSR with adjusted null hypothesis
        adjusted_SR0 = self.SR0 + E_max_SR
        dsr = probabilistic_sharpe_ratio(
            best_sharpe, SR0=adjusted_SR0, T=T_validation, 
            gamma3=gamma3, gamma4=gamma4, K=K
        )
        
        # Observed False Discovery Rate
        ofdr_value = oFDR(
            best_sharpe, SR0=adjusted_SR0, SR1=self.SR1 + adjusted_SR0, 
            T=T_validation, p_H1=self.p_H1, gamma3=gamma3, gamma4=gamma4, K=K
        )
        
        return {
            'raw_sharpe': best_sharpe,
            'expected_max_sr': E_max_SR,
            'deflated_sharpe_ratio': dsr,
            'observed_fdr': ofdr_value,
            'gamma3': gamma3,
            'gamma4': gamma4,
            'n_trials': K,
            'significant': dsr > 0.95 and ofdr_value < 0.25
        }

    def _train_lightgbm(self, train_df):
        """Train LightGBM ranker with Optuna optimization using CPCV.
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Full training data (CV will split internally)
            
        Returns
        -------
        callable
            Optuna objective function
            
        Notes
        -----
        This method now uses CombinatorialPurgedCV for robust validation.
        CV scoring evaluates both NDCG (for optimization) and Sharpe ratio
        (for statistical validation).
        """
        # Compute max label from training data
        target_col = self.data_processor_.target_col
        max_label = int(train_df[target_col].max())
        
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'num_leaves': trial.suggest_int('num_leaves', 20, 50),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
            
            # Compute CV scores across all folds
            cv_results = self._compute_cv_score('lgbm', params, train_df, max_label)
            
            # Store daily returns for statistical validation
            if 'daily_spread_returns' in cv_results and len(cv_results['daily_spread_returns']) > 0:
                self.validation_returns_.append(cv_results['daily_spread_returns'])
            
            # Store metrics for statistical analysis
            trial.set_user_attr('sharpe', cv_results['sharpe_mean'])
            trial.set_user_attr('sharpe_diff', cv_results['sharpe_diff_mean'])
            trial.set_user_attr('sharpe_std', cv_results['sharpe_std'])
            trial.set_user_attr('ndcg_std', cv_results['ndcg_std'])
            trial.set_user_attr('n_paths', self.n_paths_tested_)
            
            # Track for selection bias correction
            self.trial_sharpes_.append(cv_results['sharpe_mean'])
            self.trial_ndcgs_.append(cv_results['ndcg_mean'])
            
            # Return both NDCG and Sharpe Difference for multi-objective optimization
            # NDCG (Maximize), Sharpe Difference (Minimize)
            return cv_results['ndcg_mean'], cv_results['sharpe_diff_mean']
        
        return objective
    
    def _train_xgboost(self, train_df):
        """Train XGBoost ranker with Optuna optimization using CPCV.
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Full training data (CV will split internally)
            
        Returns
        -------
        callable
            Optuna objective function
            
        Notes
        -----
        This method now uses CombinatorialPurgedCV for robust validation.
        """
        # Compute max label from training data
        target_col = self.data_processor_.target_col
        max_label = int(train_df[target_col].max())
        
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 1e-8, 10.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
            
            # Compute CV scores across all folds
            cv_results = self._compute_cv_score('xgb', params, train_df, max_label)
            
            # Store daily returns for statistical validation
            if 'daily_spread_returns' in cv_results and len(cv_results['daily_spread_returns']) > 0:
                self.validation_returns_.append(cv_results['daily_spread_returns'])
            
            # Store metrics for statistical analysis
            trial.set_user_attr('sharpe', cv_results['sharpe_mean'])
            trial.set_user_attr('sharpe_diff', cv_results['sharpe_diff_mean'])
            trial.set_user_attr('sharpe_std', cv_results['sharpe_std'])
            trial.set_user_attr('ndcg_std', cv_results['ndcg_std'])
            trial.set_user_attr('n_paths', self.n_paths_tested_)
            
            # Track for selection bias correction
            self.trial_sharpes_.append(cv_results['sharpe_mean'])
            self.trial_ndcgs_.append(cv_results['ndcg_mean'])
            
            # Return both NDCG and Sharpe Difference for multi-objective optimization
            # NDCG (Maximize), Sharpe Difference (Minimize)
            return cv_results['ndcg_mean'], cv_results['sharpe_diff_mean']
        
        return objective
    
    def _train_catboost(self, train_df):
        """Train CatBoost ranker with Optuna optimization using CPCV.
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Full training data (CV will split internally)
            
        Returns
        -------
        callable
            Optuna objective function
            
        Notes
        -----
        This method now uses CombinatorialPurgedCV for robust validation.
        """
        # Compute max label from training data
        target_col = self.data_processor_.target_col
        max_label = int(train_df[target_col].max())
        
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'iterations': trial.suggest_int('iterations', 100, 1000, step=100),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
                'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
            }
            
            # Compute CV scores across all folds
            cv_results = self._compute_cv_score('catboost', params, train_df, max_label)
            
            # Store daily returns for statistical validation
            if 'daily_spread_returns' in cv_results and len(cv_results['daily_spread_returns']) > 0:
                self.validation_returns_.append(cv_results['daily_spread_returns'])
            
            # Store metrics for statistical analysis
            trial.set_user_attr('sharpe', cv_results['sharpe_mean'])
            trial.set_user_attr('sharpe_diff', cv_results['sharpe_diff_mean'])
            trial.set_user_attr('sharpe_std', cv_results['sharpe_std'])
            trial.set_user_attr('ndcg_std', cv_results['ndcg_std'])
            trial.set_user_attr('n_paths', self.n_paths_tested_)
            
            # Track for selection bias correction
            self.trial_sharpes_.append(cv_results['sharpe_mean'])
            self.trial_ndcgs_.append(cv_results['ndcg_mean'])
            
            # Return both NDCG and Sharpe Difference for multi-objective optimization
            # NDCG (Maximize), Sharpe Difference (Minimize)
            return cv_results['ndcg_mean'], cv_results['sharpe_diff_mean']
        
        return objective
    
    def fit(self, X, X_val=None):
        """
        Train all ranking models using CPCV and Optuna optimization.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training data with 'Date', 'Rank', 'Target' columns
        X_val : pd.DataFrame, optional
            Validation data for final holdout evaluation (not used in CV)
            
        Returns
        -------
        self
            Fitted ModelSelector instance
            
        Notes
        -----
        This method:
        1. Uses CombinatorialPurgedCV internally for robust validation
        2. Optimizes each model (LightGBM, XGBoost, CatBoost) with Optuna
        3. Tracks Sharpe ratios across trials for selection bias correction
        4. Applies López de Prado's statistical validation framework
        5. Reports Deflated Sharpe Ratio and oFDR for significance testing
        """
        # Reset trial tracking ONLY once at the start of fit
        # Selection bias is cumulative across ALL tested models/configurations
        self.trial_sharpes_ = []
        self.trial_ndcgs_ = []
        self.model_results_ = {}
        self.best_model_ = None
        self.best_params_ = None
        self.best_score_ = -float('inf')  # Tracks best NDCG
        self.best_sharpe_diff_ = float('inf')  # Tracks best (lowest) Sharpe Difference
        self.best_model_name_ = None
        
        train_df = X.copy()
        val_df = X_val.copy() if X_val is not None else None
        
        # Train each model with CPCV + Optuna
        models = {
            'LightGBM': self._train_lightgbm(train_df),
            'XGBoost': self._train_xgboost(train_df),
            'CatBoost': self._train_catboost(train_df)
        }
        
        for model_name, objective in models.items():
            print(f"\n{'='*60}")
            print(f"Training {model_name} with CPCV + Optuna")
            print(f"Cumulative trials so far: {len(self.trial_sharpes_)}")
            print(f"{'='*60}")
            
            try:
                # Create Optuna study with multi-objective optimization
                # Objective 1: NDCG (maximize)
                # Objective 2: Sharpe Difference (minimize)
                study = optuna.create_study(
                    directions=['maximize', 'minimize'],
                    sampler=optuna.samplers.TPESampler(seed=self.seed),
                    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
                )
                
                # Add MinTRL-based early stopping
                mintrl_callback = MinTRLCallback(
                    selector=self,
                    SR0=self.SR0, 
                    alpha=0.05, 
                    significance_threshold=0.95,
                    min_trials=self.min_trials
                )
                
                # Optimize
                study.optimize(
                    objective, 
                    n_trials=self.n_trials,
                    callbacks=[mintrl_callback],
                    show_progress_bar=True
                )
                
                # For multi-objective optimization, select best trial
                # We prioritize minimizing Sharpe Difference while ensuring reasonable NDCG
                best_trials = study.best_trials
                
                # Select trial with best (lowest) Sharpe Difference among those with NDCG > 0.85
                best_trial = None
                best_sharpe_diff = float('inf')
                for trial in best_trials:
                    ndcg_val = trial.values[0]  # First objective: NDCG
                    sharpe_diff_val = trial.values[1]  # Second objective: Sharpe Difference
                    # Prioritize minimizing Sharpe Difference but require minimum NDCG threshold
                    if ndcg_val > 0.85 and sharpe_diff_val < best_sharpe_diff:
                        best_sharpe_diff = sharpe_diff_val
                        best_trial = trial
                
                # Fallback: if no trial meets threshold, pick best (lowest) Sharpe Difference overall
                if best_trial is None:
                    best_trial = min(best_trials, key=lambda t: t.values[1])
                
                best_ndcg = best_trial.values[0]
                best_sharpe_diff = best_trial.values[1]
                # Note: best_sharpe from user attributes is the PREDICTED sharpe (profitability)
                best_sharpe_pred = best_trial.user_attrs.get('sharpe', 0.0)
                
                # Store results
                self.model_results_[model_name] = {
                    'study': study,
                    'best_params': best_trial.params,
                    'best_ndcg': best_ndcg,
                    'best_sharpe_diff': best_sharpe_diff,
                    'best_sharpe': best_sharpe_pred,
                    'n_trials_completed': len(study.trials),
                    'n_pareto_trials': len(best_trials)
                }
                
                print(f"\nBest NDCG: {best_ndcg:.4f}")
                print(f"Predicted Sharpe: {best_sharpe_pred:.4f}")
                print(f"Trials completed: {len(study.trials)}/{self.n_trials}")
                print(f"Best No. of Trials: {len(best_trials)}")
                
                # Update best model if current has better (lower) Sharpe Difference
                if best_sharpe_diff < self.best_sharpe_diff_:
                    self.best_score_ = best_ndcg  # Store NDCG as the primary score
                    self.best_sharpe_diff_ = best_sharpe_diff  # Track Sharpe Difference separately
                    self.best_params_ = best_trial.params
                    self.best_model_name_ = model_name
                    
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                traceback.print_exc()
        
        # Select best model
        if self.best_model_name_ is None:
            raise ValueError("No models were successfully trained")
            
        best_result = self.model_results_[self.best_model_name_]
        
        # Compute selection bias adjusted metrics
        if val_df is not None:
            T_validation = len(val_df['Date'].unique())
        else:
            T_validation = len(train_df['Date'].unique()) // self.n_folds
        
        # Get validation returns for statistical validation
        if len(self.validation_returns_) > 0:
            validation_returns = self.validation_returns_[-1]  # Use most recent
        else:
            validation_returns = np.array([])
        
        # Validation checks
        K = len(self.trial_sharpes_)
        assert T_validation > 0, "T_validation must be positive (number of validation days)"
        assert K > 0, "K must be positive (number of trials)"
        if len(validation_returns) > 0:
            # Allow some tolerance for aggregated returns across folds
            assert len(validation_returns) >= T_validation * 0.5, \
                f"Returns length ({len(validation_returns)}) should be close to T ({T_validation})"
        
        adjusted_metrics = self._compute_selection_bias_adjusted_metrics(
            self.trial_sharpes_, 
            best_result['best_sharpe'],
            T_validation,
            validation_returns
        )
        
        # Print statistical report
        print(f"\n{'='*60}")
        print(f"STATISTICAL VALIDATION REPORT")
        print(f"{'='*60}")
        print(f"Best Model: {self.best_model_name_}")
        print(f"Best NDCG: {best_result['best_ndcg']:.4f}")
        print(f"Best Sharpe Difference (minimized): {self.best_score_:.4f}")
        print(f"Predicted Sharpe Ratio (Daily): {best_result['best_sharpe']:.4f}")
        
        # Calculate annualized Sharpe Ratio
        annualized_sharpe = best_result['best_sharpe'] * np.sqrt(252)
        print(f"Annualized Sharpe Ratio: {annualized_sharpe:.4f}")
        
        # Leakage warning if annualized Sharpe is suspiciously high
        if annualized_sharpe > 3.0:
            print(f"\n⚠️  LEAKAGE WARNING: Annualized SR ({annualized_sharpe:.2f}) is institutionally impossible!")
            print(f"    This indicates the competition's target data contains inherent lookahead bias.")
            print(f"    Even with CPCV purging/embargoing, the provided 'Target' column uses future data.")
            print(f"    Results are valid for competition ranking but NOT for live trading.\n")
        
        print(f"Raw Sharpe Ratio: {adjusted_metrics['raw_sharpe']:.4f}")
        print(f"Expected Max SR (haircut): {adjusted_metrics['expected_max_sr']:.4f}")
        print(f"Deflated Sharpe Ratio (DSR): {adjusted_metrics['deflated_sharpe_ratio']:.4f}")
        print(f"Observed FDR: {adjusted_metrics['observed_fdr']:.4f}")
        print(f"Skewness (γ₃): {adjusted_metrics['gamma3']:.4f}")
        print(f"Kurtosis (γ₄): {adjusted_metrics['gamma4']:.4f}")
        print(f"Total trials: {adjusted_metrics['n_trials']}")
        print(f"Statistically significant: {adjusted_metrics['significant']}")
        print(f"{'='*60}\n")
        
        self.adjusted_metrics_ = adjusted_metrics
        
        # Train final model on full training data with best params
        print(f"Training final {self.best_model_name_} model on full data...")
        self._train_final_model(train_df, val_df)
        
        return self
    
    def _train_final_model(self, train_df, val_df):
        """Train the final model with best parameters on full training data."""
        target_col = self.data_processor_.target_col
        max_label = int(train_df[target_col].max())
        
        if self.best_model_name_ == 'LightGBM':
            train_data, label_gain = self.data_processor_.prepare_lgb_dataset(train_df, max_label=max_label)
            X_train, y_train = train_data.data, train_data.label
            group_train = train_data.get_group()
            cat_features = [col for col in self.data_processor_.cat_features if col in X_train.columns]
            
            self.best_model_ = lgb.LGBMRanker(
                objective='lambdarank',
                metric=self.eval_metric,
                device='cpu', # GPU is not supported for large bin sizes
                n_jobs=-1,
                random_state=self.seed,
                verbose=-1,
                label_gain=label_gain,  # Pass label_gain to model, not Dataset
                **self.best_params_
            )
            
            if val_df is not None:
                val_data, _ = self.data_processor_.prepare_lgb_dataset(val_df, reference=train_data, max_label=max_label)
                X_val, y_val = val_data.data, val_data.label
                group_val = val_data.get_group()
                self.best_model_.fit(
                    X_train, y_train, group=group_train,
                    eval_set=[(X_val, y_val)], eval_group=[group_val],
                    eval_metric=self.eval_metric, eval_at=self.eval_at,
                    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
                    categorical_feature=cat_features
                )
            else:
                self.best_model_.fit(X_train, y_train, group=group_train, categorical_feature=cat_features)
                
        elif self.best_model_name_ == 'XGBoost':
            X_train, y_train, qid_train = self.data_processor_.prepare_xgb_data(train_df, max_label=max_label)
            
            self.best_model_ = xgb.XGBRanker(
                objective='rank:ndcg',
                tree_method='hist' if self.device.lower() == 'gpu' else 'auto',
                n_jobs=-1,
                random_state=self.seed,
                ndcg_exp_gain=False,
                early_stopping_rounds=30 if val_df is not None else None,
                **self.best_params_
            )
            
            if val_df is not None:
                X_val, y_val, qid_val = self.data_processor_.prepare_xgb_data(val_df, max_label=max_label)
                # Note: qid, eval_qid, verbose are valid XGBRanker.fit() parameters (XGBoost 3.x)
                # Linter errors are false positives from outdated type stubs
                self.best_model_.fit( # type: ignore # noqa: F821
                    X_train, y_train, qid=qid_train,
                    eval_set=[(X_val, y_val)], eval_qid=[qid_val],
                    verbose=False # type: ignore # noqa: F821
                )
            else:
                self.best_model_.fit(X_train, y_train, qid=qid_train, verbose=False) # type: ignore # noqa: F821
        elif self.best_model_name_ == 'CatBoost':
            train_pool = self.data_processor_.prepare_catboost_pool(train_df, max_label=max_label)
            ndcg_metric = f'NDCG:top={self.eval_at[-1]}'
            
            self.best_model_ = cb.CatBoostRanker(
                loss_function='YetiRank',
                eval_metric=ndcg_metric,
                task_type=self.device.upper(),
                random_seed=self.seed,
                verbose=False,
                **self.best_params_
            )
            
            if val_df is not None:
                val_pool = self.data_processor_.prepare_catboost_pool(val_df, max_label=max_label)
                # Note: When X is a Pool, y is not required; early_stopping_rounds and verbose are valid
                # Linter errors are false positives from type stubs
                self.best_model_.fit(train_pool, eval_set=val_pool, early_stopping_rounds=30, verbose=False) # type: ignore # noqa: F821
            else:
                self.best_model_.fit(train_pool, verbose=False) # type: ignore # noqa: F821
    
    def predict(self, X):
        """
        Predict rankings using the best model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features for prediction.
            
        Returns
        -------
        np.ndarray
            Predicted ranking scores.
        """
        if self.best_model_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        X_pred = X.copy()
        
        if self.best_model_name_ == 'LightGBM':
            # Use data processor but preserve the internal index for alignment.
            # We sort to ensure deterministic categorical features, then predict.
            X_sorted = X_pred.sort_values(by=[self.data_processor_.group_col, 'SecuritiesCode'])
            
            # Use prepare_lgb_dataset on the sorted data
            lgb_data = self.data_processor_.prepare_lgb_dataset(X_sorted)[0].data
            preds_sorted = self.best_model_.predict(lgb_data)
            
            # Map sorted predictions back to the original input index to ensure alignment
            return pd.Series(preds_sorted, index=X_sorted.index).reindex(X.index).values
            
        elif self.best_model_name_ == 'XGBoost':
            # Similar fix for XGBoost: sort for feature consistency, predict, then reindex
            X_sorted = X_pred.sort_values(by=[self.data_processor_.group_col, 'SecuritiesCode'])
            X_feat = X_sorted.drop(columns=['Date', 'Rank', 'Target'], errors='ignore')
            
            # Enforce exact feature set and order from training
            if hasattr(self.data_processor_, 'feature_names_') and self.data_processor_.feature_names_:
                X_feat = X_feat[self.data_processor_.feature_names_]
                
            # Handle categorical features (must match training logic)
            for cat_col in self.data_processor_.cat_features:
                if cat_col in X_feat.columns:
                    X_feat[cat_col] = pd.Categorical(X_feat[cat_col]).codes

            preds_sorted = self.best_model_.predict(X_feat)
            return pd.Series(preds_sorted, index=X_sorted.index).reindex(X.index).values

        elif self.best_model_name_ == 'CatBoost':
            # Similar fix for CatBoost
            X_sorted = X_pred.sort_values(by=[self.data_processor_.group_col, 'SecuritiesCode'])
            X_feat = X_sorted.drop(columns=['Date', 'Rank', 'Target'], errors='ignore')
            
            # Enforce exact feature set and order from training
            if hasattr(self.data_processor_, 'feature_names_') and self.data_processor_.feature_names_:
                X_feat = X_feat[self.data_processor_.feature_names_]
            
            # Convert categorical features to string
            for cat_col in self.data_processor_.cat_features:
                if cat_col in X_feat.columns:
                    X_feat[cat_col] = X_feat[cat_col].astype(str)

            preds_sorted = self.best_model_.predict(X_feat)
            return pd.Series(preds_sorted, index=X_sorted.index).reindex(X.index).values
        
        else:
            raise ValueError(f"Unknown model name: {self.best_model_name_}")