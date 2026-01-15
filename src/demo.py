"""
Demo script for ensemble house price predictor.
Automatically finds models and loads ensemble_model module.
"""

import pickle
import numpy as np
import os
import sys


def find_project_root():
    """Find project root by looking for models/ directory."""
    current = os.path.dirname(os.path.abspath(__file__))

    # Check if we're in src/
    if os.path.basename(current) == 'src':
        project_root = os.path.dirname(current)
    else:
        project_root = current

    # Verify models/ exists
    if not os.path.exists(os.path.join(project_root, 'models')):
        raise FileNotFoundError(f"Cannot find models/ directory. Looking in: {project_root}")

    return project_root


# Setup paths
PROJECT_ROOT = find_project_root()
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)

# Import CalibratedPredictor from ensemble_model.py
from ensemble_model import CalibratedPredictor

# Load model
model_path = os.path.join(PROJECT_ROOT, 'models', 'ensemble_production_v1.pkl')

print("=" * 60)
print("ENSEMBLE HOUSE PRICE PREDICTOR - DEMO")
print("=" * 60)
print(f"\nProject Root: {PROJECT_ROOT}")
print(f"Model Path:   {model_path}")

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("✓ Model loaded successfully")
except FileNotFoundError:
    print(f"✗ Model file not found at: {model_path}")
    print("\nPlease ensure the model was saved correctly.")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error loading model: {e}")
    sys.exit(1)

# Display model info
print(f"\nModel Metadata:")
print(f"  Version:  {model['metadata']['version']}")
print(f"  Test R²:  {model['metadata']['test_r2']}")
print(f"  Coverage: {model['metadata']['coverage']}%")
print(f"  Created:  {model['metadata']['created']}")
print(f"\nFeatures ({len(model['feature_names'])}):")
for i, feat in enumerate(model['feature_names'], 1):
    print(f"  {i}. {feat}")

# Example predictions
print("\n" + "=" * 60)
print("EXAMPLE PREDICTIONS")
print("=" * 60)

examples = [
    {
        'name': 'Luxury Home',
        'features': [9, 2500, 3, 1500, 1500, 2015],
        'description': 'High quality, large, new'
    },
    {
        'name': 'Average Home',
        'features': [6, 1500, 2, 1000, 1000, 1990],
        'description': 'Standard quality, medium size'
    },
    {
        'name': 'Budget Home',
        'features': [4, 900, 1, 500, 500, 1960],
        'description': 'Lower quality, small, old'
    }
]

for example in examples:
    print(f"\n{example['name']} ({example['description']}):")
    print("-" * 60)

    # Format input
    house = np.array([example['features']])

    feature_labels = ['OverallQual', 'GrLivArea', 'GarageCars',
                      'TotalBsmtSF', '1stFlrSF', 'YearBuilt']

    for label, value in zip(feature_labels, example['features']):
        unit = 'sqft' if 'Area' in label or 'SF' in label else ''
        print(f"  {label:12s}: {value:4d} {unit}")

    # Standardize using saved parameters
    house_scaled = (house - model['scaler_params']['mean']) / model['scaler_params']['std']

    # Predict
    try:
        result = model['calibrated_predictor'].predict(house_scaled)

        print(f"\n  📊 Results:")
        print(f"     Predicted:  ${result['price']:>10,.0f}")
        print(f"     Confidence: {result['confidence'].upper():>10}")
        print(f"     Range:      ${result['range'][0]:>10,.0f} - ${result['range'][1]:>10,.0f}")
        print(f"     Agreement:  {(1 - result['disagreement']) * 100:>9.1f}%")

    except Exception as e:
        print(f"  ✗ Prediction failed: {e}")

print("\n" + "=" * 60)
print("✓ Demo complete")
print("=" * 60)