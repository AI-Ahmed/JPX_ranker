"""
Kaggle Submission Module for JPX Tokyo Stock Exchange Prediction Competition

This module provides a production-ready submission function that handles:
- Model loading
- Kaggle API integration
- Prediction generation with validation
- Comprehensive error handling and logging

Usage in Jupyter Notebook:
    from jpx_ranker.kaggle_submission import run_kaggle_submission
    
    # For Kaggle submission
    run_kaggle_submission(
        model_path='kaggle/working/best_ranker_model.pkl',
        fast_period=5,
        slow_period=10,
        enable_submission=True
    )
"""

import os
import pickle
from pathlib import Path
from typing import Optional

from loguru import logger

from ._src.infer import InferencePipeline


def run_kaggle_submission(
    model_path: str,
    fast_period: int = 5,
    slow_period: int = 10,
    enable_submission: bool = False,
    verbose: bool = True
) -> None:
    """
    Execute Kaggle submission for JPX Tokyo Stock Exchange Prediction competition.
    
    This function handles the complete submission workflow including model loading,
    Kaggle API integration, prediction generation, validation, and submission.
    
    Parameters
    ----------
    model_path : str
        Path to the saved trained model (pickle file).
    fast_period : int, optional
        Fast moving average period for feature engineering (default: 5).
    slow_period : int, optional
        Slow moving average period for feature engineering (default: 10).
    enable_submission : bool, optional
        If True, actually submit to Kaggle. If False, run in test mode (default: False).
    verbose : bool, optional
        If True, print detailed progress information (default: True).
        
    Returns
    -------
    None
    
    Raises
    ------
    ImportError
        If jpx_tokyo_market_prediction module is not available (not in Kaggle environment).
    FileNotFoundError
        If the model file does not exist at the specified path.
    AssertionError
        If prediction validation fails.
    Exception
        For any other errors during submission.
        
    Notes
    -----
    This function is designed to be called from a Jupyter notebook cell.
    For local testing, set `enable_submission=False`.
    
    The function performs the following steps:
    1. Initialize Kaggle environment
    2. Load trained model
    3. Initialize inference pipeline
    4. Iterate through test data
    5. Generate and validate predictions
    6. Submit to Kaggle
    
    Examples
    --------
    In Kaggle notebook:
    >>> run_kaggle_submission(
    ...     model_path='kaggle/working/best_ranker_model.pkl',
    ...     fast_period=5,
    ...     slow_period=10,
    ...     enable_submission=True
    ... )
    
    For local testing:
    >>> run_kaggle_submission(
    ...     model_path='models/best_ranker_model.pkl',
    ...     enable_submission=False
    ... )
    """
    
    if not enable_submission:
        # Local testing mode
        print("=" * 80)
        print("KAGGLE SUBMISSION: DISABLED (Local Testing Mode)")
        print("=" * 80)
        print("")
        print("To enable Kaggle submission:")
        print("  1. Set enable_submission=True")
        print(f"  2. Ensure the model is saved at: {model_path}")
        print("  3. Run this in the Kaggle notebook environment")
        print("")
        print("For local testing, use the InferencePipeline class directly:")
        print("  >>> from jpx_ranker._src.infer import InferencePipeline")
        print("  >>> inference_pipeline = InferencePipeline(model=trained_model)")
        print("  >>> predictions = inference_pipeline.predict(test_data)")
        print("=" * 80)
        return
    
    try:
        import jpx_tokyo_market_prediction
        
        if verbose:
            logger.info("=" * 80)
            logger.info("STARTING KAGGLE SUBMISSION")
            logger.info("=" * 80)
        
        # ========================================================================
        # 1. Initialize Kaggle Environment
        # ========================================================================
        if verbose:
            logger.info("Initializing Kaggle environment...")
        env = jpx_tokyo_market_prediction.make_env()
        iter_test = env.iter_test()
        if verbose:
            logger.success("✓ Kaggle environment initialized")
        
        # ========================================================================
        # 2. Load Trained Model
        # ========================================================================
        model_file = Path(model_path)
        if verbose:
            logger.info(f"Loading trained model from: {model_file}")
        
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model file not found at {model_file}. "
                f"Please ensure the model was saved during training."
            )
        
        with open(model_file, 'rb') as f:
            trained_model = pickle.load(f)
        
        if verbose:
            model_size_kb = model_file.stat().st_size / 1024
            logger.success(f"✓ Model loaded successfully ({model_size_kb:.2f} KB)")
        
        # ========================================================================
        # 3. Initialize Inference Pipeline
        # ========================================================================
        if verbose:
            logger.info("Initializing inference pipeline...")
        
        inference_pipeline = InferencePipeline(
            model=trained_model,
            fast_period=fast_period,
            slow_period=slow_period
        )
        
        if verbose:
            logger.success(
                f"✓ Inference pipeline initialized "
                f"(fast_period={fast_period}, slow_period={slow_period})"
            )
        
        # ========================================================================
        # 4. Iterate Through Test Data and Make Predictions
        # ========================================================================
        if verbose:
            logger.info("Starting prediction loop...")
        
        prediction_count = 0
        
        for test_batch in iter_test:
            prices, options, financials, trades, secondary_prices, sample_prediction = test_batch
            
            try:
                # Make predictions using the inference pipeline
                predictions = inference_pipeline.predict_for_kaggle(
                    prices, 
                    sample_prediction
                )
                
                # ============================================================
                # Validate Predictions
                # ============================================================
                # These assertions ensure Kaggle submission format compliance
                assert 'Rank' in predictions.columns, \
                    "Predictions must have 'Rank' column"
                assert len(predictions) == len(sample_prediction), \
                    f"Prediction count mismatch: {len(predictions)} != {len(sample_prediction)}"
                assert predictions['Rank'].min() >= 0, \
                    f"Invalid rank: minimum rank is {predictions['Rank'].min()}, must be >= 0"
                assert predictions['Rank'].max() < len(predictions), \
                    f"Invalid rank: maximum rank is {predictions['Rank'].max()}, must be < {len(predictions)}"
                assert not predictions['Rank'].duplicated().any(), \
                    "Duplicate ranks detected - all ranks must be unique"
                
                # Submit predictions to Kaggle
                env.predict(predictions)
                
                prediction_count += 1
                if verbose and prediction_count % 10 == 0:
                    logger.info(f"Processed {prediction_count} prediction batches...")
                
            except AssertionError as ae:
                logger.error(f"Validation error in batch {prediction_count + 1}: {ae}")
                logger.error(f"Predictions shape: {predictions.shape}")
                logger.error(f"Sample prediction shape: {sample_prediction.shape}")
                raise
            
            except Exception as e:
                logger.error(f"Prediction error in batch {prediction_count + 1}: {e}")
                logger.error(f"Prices shape: {prices.shape if hasattr(prices, 'shape') else 'N/A'}")
                raise
        
        # ========================================================================
        # 5. Submission Complete
        # ========================================================================
        if verbose:
            logger.info("=" * 80)
            logger.success("✓ KAGGLE SUBMISSION COMPLETED SUCCESSFULLY!")
            logger.info(f"  Total prediction batches: {prediction_count}")
            logger.info("=" * 80)
        
    except ImportError as ie:
        logger.error(
            f"Failed to import jpx_tokyo_market_prediction: {ie}\n"
            f"This module is only available in the Kaggle environment. "
            f"Set enable_submission=False for local testing."
        )
        raise
    
    except FileNotFoundError as fnf:
        logger.error(f"Model file error: {fnf}")
        logger.error(
            f"Please ensure the training pipeline completed successfully "
            f"and saved the model to: {model_path}"
        )
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error during Kaggle submission: {e}")
        logger.exception("Full traceback:")
        raise