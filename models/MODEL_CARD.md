# Ensemble House Price Predictor - Model Card

## Model Details
- **Version**: 1.0
- **Date**: January 15, 2026
- **Type**: Ensemble (Linear Regression + XGBoost)
- **Task**: Regression (house price prediction)

## Performance
- **Test R²**: 0.8833
- **RMSE**: 0.1416 (log scale)
- **Coverage**: 88.4% (confidence intervals)

## Intended Use
- Portfolio valuation
- Automated property appraisals
- Risk assessment for lending

## Limitations
- Training data: Ames, Iowa (2006-2010)
- Struggles with extreme outliers (<$50k)
- No temporal/seasonal features

## Ethical Considerations
- Do not use for discriminatory pricing
- Requires human review for low-confidence predictions
- May not generalize to other geographic markets
