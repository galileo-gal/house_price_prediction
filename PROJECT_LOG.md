# House Price Predictor - Project Log

## Project Overview
Building a house price prediction model from scratch using linear regression and its variants.
- **Dataset**: Kaggle House Prices (1458 samples, 81 features)
- **Goal**: Implement ML algorithms from scratch, understand fundamentals
- **Timeline**: 2 weeks (Days 1-5 complete)

---

## Day 1-2: Data Preparation & EDA
**Date**: [Your dates]

### Tasks Completed
- ✅ Loaded and explored dataset (1460 samples, 81 features)
- ✅ Missing value analysis (19 features with missing data)
- ✅ Correlation analysis with SalePrice
- ✅ Feature selection: 7 features chosen based on correlation
  - OverallQual (0.79), GrLivArea (0.71), GarageCars (0.64)
  - TotalBsmtSF (0.61), 1stFlrSF (0.61), FullBath (0.56), YearBuilt (0.52)
- ✅ Outlier detection and removal (2 outliers removed)
- ✅ Log transformation applied to SalePrice (skewness: 1.88 → 0.12)
- ✅ Train/Val/Test split: 75%/15%/10% (1092/220/146 samples)
- ✅ Feature standardization (mean=0, std=1)

### Key Findings
- Right-skewed SalePrice distribution required log transformation
- Strong multicollinearity between some features (e.g., GarageArea vs GarageCars)
- 2 extreme outliers identified (large houses with unusually low prices)

### Files
- `notebooks/01_eda.ipynb`
- `data/train.csv` (gitignored)

---

## Day 3: Linear Regression from Scratch
**Date**: [Your date]

### Tasks Completed
- ✅ Implemented `LinearRegressionScratch` class with gradient descent
- ✅ Training: lr=0.01, 1000 iterations
- ✅ Cost reduction: 99.98% (converged successfully)
- ✅ Model evaluation and visualization

### Results
```
Training R²:   0.8449
Validation R²: 0.8484
Gap:           0.0035
```
- ✅ Excellent generalization - no overfitting
- ✅ Validation performs slightly better than training

### Files
- `src/linear_regression.py` (LinearRegressionScratch class)

---

## Day 4: Polynomial Regression
**Date**: [Your date]

### Tasks Completed
- ✅ Implemented `PolynomialRegressionScratch` (Normal Equation)
- ✅ Implemented `PolynomialRegressionGradientDescent` (GD with lr=0.06, 2000 iter)
- ✅ Debugged gradient descent for high-dimensional features (35 features)
- ✅ Tested polynomial degrees 1-5
- ✅ BIC/AIC analysis for model selection
- ✅ Comparative visualizations

### Results
| Degree | Features | BIC      | AIC      | Val R²  |
|--------|----------|----------|----------|---------|
| 1      | 7        | -4023.50 | -4063.46 | 0.8484  |
| 2      | 35       | -3920.13 | -4099.98 | 0.8420  |
| 3      | 119      | -3452.07 | -4051.56 | 0.8333  |
| 4+     | 329+     | Exploded | Exploded | Negative|

### Key Learnings
- Polynomial features don't improve performance for this dataset
- Linear model (degree=1) is optimal by all metrics
- High-degree polynomials (4+) catastrophically overfit
- Gradient descent required careful learning rate tuning (lr=0.06 for degree=2)

### Files
- `src/linear_regression.py` (added Polynomial classes)

---

## Day 5: Regularization (Ridge & Lasso)
**Date**: [Your date]

### Tasks Completed
- ✅ Implemented `RidgeRegressionScratch` (L2 regularization, Normal Equation)
- ✅ Implemented `LassoRegressionScratch` (L1 regularization, Coordinate Descent)
- ✅ Tested multiple alpha values for both methods
- ✅ Feature selection analysis with Lasso
- ✅ Comparative visualizations

### Ridge Results
```
Best alpha: 10.0
Training R²:   0.8450
Validation R²: 0.8476
```
- No improvement over linear model

### Lasso Results
```
Best alpha: 0.0001
Training R²:   0.8451
Validation R²: 0.8475
Features:      7/7 (no feature elimination until alpha=10)
```
- No improvement over linear model
- Minimal feature selection even at high alpha

### Key Findings
- **Regularization doesn't help** - model is not overfitting
- Linear model with 7 features is already well-regularized
- Ridge and Lasso performance nearly identical to baseline
- **Conclusion**: Simple linear regression is optimal for this dataset

### Files
- `src/linear_regression.py` (added Ridge and Lasso classes)

---

## Summary: Days 1-5

### Best Model
**Linear Regression (No Regularization)**
- Training R²: 0.8449
- Validation R²: 0.8484
- Features: 7
- Method: Gradient Descent

### All Models Comparison
| Model               | Train R² | Val R²  | Notes                    |
|---------------------|----------|---------|--------------------------|
| Linear Regression   | 0.8449   | 0.8484  | ✅ Best - Simple & Effective |
| Polynomial (deg=2)  | 0.8575   | 0.8420  | Slight overfitting       |
| Ridge (α=10)        | 0.8450   | 0.8476  | No improvement           |
| Lasso (α=0.0001)    | 0.8451   | 0.8475  | No improvement           |

### Key Takeaways
1. **Simplicity wins**: Linear model outperforms complex variants
2. **Feature engineering matters**: 7 well-chosen features > 35 polynomial features
3. **Data quality > Model complexity**: Good preprocessing (log transform, outlier removal) more impactful than fancy algorithms
4. **Occam's Razor validated**: Don't add complexity without clear benefit

---

## Next Steps (Days 6-14)

### Day 6-7: Model Comparison & sklearn Validation
- [ ] Compare with sklearn implementations
- [ ] Cross-validation
- [ ] Hyperparameter tuning
- [ ] Test set evaluation

### Day 8-9: Error Analysis & Insights
- [ ] Residual analysis
- [ ] Feature importance visualization
- [ ] Prediction error patterns
- [ ] Model limitations

### Day 10-12: Documentation & Article
- [ ] Code refactoring and documentation
- [ ] Write Medium article explaining methodology
- [ ] Create visualizations for article
- [ ] GitHub README update

### Day 13-14: Deployment Prep
- [ ] Model serialization (pickle/joblib)
- [ ] Create prediction pipeline
- [ ] Simple Flask/FastAPI demo
- [ ] Docker containerization (optional)

---

## Technical Specifications

### Environment
- Python 3.12
- Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
- IDE: PyCharm + Jupyter Lab
- Version Control: Git/GitHub

### Repository Structure
```
house-price-predictor/
├── data/
│   ├── train.csv (gitignored)
│   └── test.csv (gitignored)
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   └── linear_regression.py
├── .gitignore
├── requirements.txt
├── README.md
└── PROJECT_LOG.md
```

### Classes Implemented
1. `LinearRegressionScratch` - Gradient descent
2. `PolynomialRegressionScratch` - Normal equation
3. `PolynomialRegressionGradientDescent` - Gradient descent for polynomials
4. `RidgeRegressionScratch` - L2 regularization
5. `LassoRegressionScratch` - L1 regularization with coordinate descent

---

## Lessons Learned

### Technical
- Gradient descent requires careful learning rate tuning for high dimensions
- Normal equation is faster and exact for small-medium datasets
- Feature scaling is critical for gradient descent convergence
- BIC/AIC are excellent tools for model selection

### Practical
- Exploratory data analysis is 50% of the work
- Simple models are easier to debug and interpret
- Validation metrics more important than training metrics
- Occam's Razor: prefer simpler models when performance is equal

---

**Last Updated**: [12-1-26]
**Status**: Days 1-5 Complete ✅ | Ready for Day 6