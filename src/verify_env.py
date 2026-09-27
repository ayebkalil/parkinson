import sys
import pandas as pd
import numpy as np
import sklearn
import xgboost
import lightgbm
import pyarrow

print("=== Python Environment Verification ===")
print("Python:", sys.version.split()[0])
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Scikit-learn:", sklearn.__version__)
print("XGBoost:", xgboost.__version__)
print("LightGBM:", lightgbm.__version__)
print("PyArrow:", pyarrow.__version__)

try:
    import torch
    print("PyTorch:", torch.__version__)
except ImportError:
    print("PyTorch: Not yet installed (can be installed for Phase 4)")

print("\nAll core Phase 1 & Phase 2 libraries are ready!")
