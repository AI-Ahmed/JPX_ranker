"""
Kaggle Authentication Setup Module

This module handles Kaggle authentication using environment variables
instead of interactive login.

Notes
-----
Requires the following environment variables to be set in `.env`:
- KAGGLE_USERNAME: Your Kaggle username
- KAGGLE_KEY: Your Kaggle API key (not KAGGLE_API_TOKEN)

The Kaggle API expects the key to be named `KAGGLE_KEY`, not `KAGGLE_API_TOKEN`.

All downloads are saved to the project's `kaggle/input/` directory
to maintain compatibility with Kaggle's environment structure.

This allows notebooks to work seamlessly both locally and on Kaggle.

References
----------
.. [1] Kaggle API Documentation: https://github.com/Kaggle/kaggle-api
"""

import os
import shutil
from pathlib import Path
from dotenv import load_dotenv


# Project data directory - Kaggle-compatible structure
PROJECT_ROOT = Path(__file__).parent.parent
KAGGLE_DIR = PROJECT_ROOT / 'kaggle'
INPUT_DIR = KAGGLE_DIR / 'input'


def _ensure_kaggle_directories():
    """
    Ensure Kaggle-compatible directories exist.
    
    Creates the following directory structure if not present:
    - kaggle/input/
    
    This matches Kaggle's environment structure for seamless compatibility.
    """
    INPUT_DIR.mkdir(parents=True, exist_ok=True)


def setup_kaggle_credentials():
    """
    Load Kaggle credentials from environment variables.
    
    This function loads the `.env` file and sets up Kaggle authentication
    using environment variables. The Kaggle API will automatically use
    these credentials without requiring interactive login.
    
    Returns
    -------
    bool
        True if credentials are loaded successfully, False otherwise.
    
    Raises
    ------
    ValueError
        If required environment variables are not set.
    
    Notes
    -----
    The Kaggle API looks for two environment variables:
    - KAGGLE_USERNAME: Your Kaggle username
    - KAGGLE_KEY: Your Kaggle API key
    
    If you have `KAGGLE_API_TOKEN` in your `.env` instead of `KAGGLE_KEY`,
    this function will map it correctly.
    
    Examples
    --------
    >>> from jpx_ranker.kaggle_setup import setup_kaggle_credentials
    >>> setup_kaggle_credentials()
    True
    >>> import kagglehub
    >>> # No need to call kagglehub.login() anymore!
    >>> data_path = kagglehub.competition_download('jpx-tokyo-stock-exchange-prediction')
    """
    # Load environment variables from .env file
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
    
    # Get credentials from environment
    username = os.getenv('KAGGLE_USERNAME')
    api_key = os.getenv('KAGGLE_KEY') or os.getenv('KAGGLE_API_TOKEN')
    
    if not username or not api_key:
        raise ValueError(
            "Kaggle credentials not found. Please ensure your .env file contains:\n"
            "KAGGLE_USERNAME=your_username\n"
            "KAGGLE_KEY=your_api_key"
        )
    
    # Set the credentials for Kaggle API
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = api_key
    
    print(f"✓ Kaggle credentials loaded for user: {username}")
    return True


def download_competition_data(competition_name='jpx-tokyo-stock-exchange-prediction', force_download=False):
    """
    Download competition data from Kaggle to Kaggle-compatible directory.
    
    Parameters
    ----------
    competition_name : str, optional
        Name of the Kaggle competition, by default 'jpx-tokyo-stock-exchange-prediction'
    force_download : bool, optional
        If True, re-download even if cached, by default False
    
    Returns
    -------
    str
        Path to the downloaded competition data (e.g., '/kaggle/input/jpx-tokyo-stock-exchange-prediction')
    
    Raises
    ------
    Exception
        If you haven't accepted the competition rules or authentication fails
    
    Notes
    -----
    Before downloading competition data, you must:
    1. Visit the competition page
    2. Accept the competition rules
    3. Ensure your API credentials are valid
    
    Data is downloaded to: `kaggle/input/<competition_name>/`
    
    This path structure matches Kaggle's environment, so notebooks work locally and on Kaggle.
    
    If you encounter data corruption errors, try setting force_download=True
    
    Examples
    --------
    >>> from jpx_ranker.kaggle_setup import download_competition_data
    >>> data_path = download_competition_data()
    >>> print(f"Data downloaded to: {data_path}")
    >>> # Use in notebook: pd.read_csv(f'{data_path}/train.csv')
    
    >>> # Force re-download if cache is corrupted
    >>> data_path = download_competition_data(force_download=True)
    """
    import kagglehub
    
    # Ensure credentials and directories are set up
    setup_kaggle_credentials()
    _ensure_kaggle_directories()
    
    # Target directory in project (Kaggle-compatible path)
    target_dir = INPUT_DIR / competition_name
    
    # Check if already exists and not forcing download
    if target_dir.exists() and not force_download:
        print(f"✓ Competition data already exists at: {target_dir}")
        print(f"  Use in notebook: '/kaggle/input/{competition_name}'")
        return str(target_dir)
    
    print(f"Downloading competition data: {competition_name}")
    if force_download:
        print("  (Force download enabled - ignoring cache)")
    
    try:
        # Download to kagglehub cache first
        cache_path = kagglehub.competition_download(competition_name, force_download=force_download)
        print(f"  Downloaded to cache: {cache_path}")
        
        # Copy to project kaggle/input directory
        if target_dir.exists():
            shutil.rmtree(target_dir)
        
        shutil.copytree(cache_path, target_dir)
        print(f"✓ Data copied to: {target_dir}")
        print(f"  Use in notebook: '/kaggle/input/{competition_name}'")
        return str(target_dir)
        
    except Exception as e:
        if "401" in str(e) or "permission" in str(e).lower():
            print("\n" + "="*70)
            print("❌ PERMISSION ERROR - ACTION REQUIRED")
            print("="*70)
            print(f"\nYou need to accept the competition rules first:")
            print(f"1. Visit: https://www.kaggle.com/competitions/{competition_name}")
            print(f"2. Click 'Join Competition' or 'I Understand and Accept'")
            print(f"3. Accept the rules")
            print(f"4. Then re-run this code\n")
            print("Also verify your credentials are correct in .env file:")
            print(f"  KAGGLE_USERNAME: {os.getenv('KAGGLE_USERNAME')}")
            print(f"  KAGGLE_KEY: {'***' + os.getenv('KAGGLE_KEY', '')[-4:] if os.getenv('KAGGLE_KEY') else 'NOT SET'}")
            print("="*70 + "\n")
        raise


def download_dataset(dataset_name, force_download=False):
    """
    Download a dataset from Kaggle to Kaggle-compatible directory.
    
    Parameters
    ----------
    dataset_name : str
        Name of the Kaggle dataset (e.g., 'dsxavier/jpx-pre')
    force_download : bool, optional
        If True, re-download even if cached, by default False
    
    Returns
    -------
    str
        Path to the downloaded dataset (e.g., '/kaggle/input/jpx-pre')
    
    Raises
    ------
    Exception
        If authentication fails or dataset is not accessible
    
    Notes
    -----
    Data is downloaded to: `kaggle/input/<dataset_short_name>/`
    
    The dataset short name is extracted from the full path (e.g., 'dsxavier/jpx-pre' -> 'jpx-pre').
    This matches how Kaggle names datasets in its environment.
    
    If you encounter data corruption errors, try setting force_download=True
    
    Examples
    --------
    >>> from jpx_ranker.kaggle_setup import download_dataset
    >>> data_path = download_dataset('dsxavier/jpx-pre')
    >>> print(f"Dataset downloaded to: {data_path}")
    >>> # Use in notebook: pd.read_csv(f'{data_path}/data.csv')
    
    >>> # Force re-download if cache is corrupted
    >>> data_path = download_dataset('dsxavier/jpx-pre', force_download=True)
    """
    import kagglehub
    
    # Ensure credentials and directories are set up
    setup_kaggle_credentials()
    _ensure_kaggle_directories()
    
    # Extract dataset short name (e.g., 'dsxavier/jpx-pre' -> 'jpx-pre')
    dataset_short_name = dataset_name.split('/')[-1]
    target_dir = INPUT_DIR / dataset_short_name
    
    # Check if already exists and not forcing download
    if target_dir.exists() and not force_download:
        print(f"✓ Dataset already exists at: {target_dir}")
        print(f"  Use in notebook: '/kaggle/input/{dataset_short_name}'")
        return str(target_dir)
    
    print(f"Downloading dataset: {dataset_name}")
    if force_download:
        print("  (Force download enabled - ignoring cache)")
    
    try:
        # Download to kagglehub cache first
        cache_path = kagglehub.dataset_download(dataset_name, force_download=force_download)
        print(f"  Downloaded to cache: {cache_path}")
        
        # Copy to project kaggle/input directory
        if target_dir.exists():
            shutil.rmtree(target_dir)
        
        shutil.copytree(cache_path, target_dir)
        print(f"✓ Data copied to: {target_dir}")
        print(f"  Use in notebook: '/kaggle/input/{dataset_short_name}'")
        return str(target_dir)
        
    except Exception as e:
        if "401" in str(e) or "permission" in str(e).lower():
            print("\n" + "="*70)
            print("❌ AUTHENTICATION ERROR")
            print("="*70)
            print(f"\nFailed to download dataset: {dataset_name}")
            print("\nVerify your credentials in .env file:")
            print(f"  KAGGLE_USERNAME: {os.getenv('KAGGLE_USERNAME')}")
            print(f"  KAGGLE_KEY: {'***' + os.getenv('KAGGLE_KEY', '')[-4:] if os.getenv('KAGGLE_KEY') else 'NOT SET'}")
            print("\nMake sure:")
            print("1. Your .env file exists in the project root")
            print("2. Credentials are correct (get them from https://www.kaggle.com/settings/account)")
            print("3. The dataset exists and is public or you have access to it")
            print("="*70 + "\n")
        raise

