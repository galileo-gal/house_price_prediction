"""
Ensemble model with calibrated confidence intervals.
"""

import numpy as np


class CalibratedPredictor:
    """
    Production ensemble predictor with confidence scoring.

    Combines Linear Regression + XGBoost predictions with
    calibrated confidence intervals based on model disagreement.
    """

    def __init__(self, linear_model, xgb_model, meta_model,
                 high_thresh, med_thresh, high_margin, med_margin, low_margin):
        """
        Parameters
        ----------
        linear_model : LinearRegressionScratch
            Base linear model
        xgb_model : XGBRegressor
            Base XGBoost model
        meta_model : Ridge
            Meta-learner for blending
        high_thresh : float
            Disagreement threshold for high confidence
        med_thresh : float
            Disagreement threshold for medium confidence
        high_margin : float
            Confidence margin for high confidence (log scale)
        med_margin : float
            Confidence margin for medium confidence (log scale)
        low_margin : float
            Confidence margin for low confidence (log scale)
        """
        self.linear = linear_model
        self.xgb = xgb_model
        self.meta = meta_model
        self.high_thresh = high_thresh
        self.med_thresh = med_thresh
        self.margins = {
            'high': high_margin,
            'medium': med_margin,
            'low': low_margin
        }

    def predict(self, X):
        """
        Predict price with confidence interval.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix (standardized)

        Returns
        -------
        dict
            - price: float
                Predicted price in dollars
            - confidence: str
                'high', 'medium', or 'low'
            - range: tuple
                (lower_bound, upper_bound) in dollars
            - disagreement: float
                Absolute difference between base models (log scale)
        """
        # Get base model predictions
        linear_pred = self.linear.predict(X)
        xgb_pred = self.xgb.predict(X)

        # Calculate disagreement
        disagreement = np.abs(linear_pred - xgb_pred)

        # Determine confidence level
        if disagreement < self.high_thresh:
            confidence = 'high'
        elif disagreement < self.med_thresh:
            confidence = 'medium'
        else:
            confidence = 'low'

        margin = self.margins[confidence]

        # Ensemble prediction via meta-model
        X_meta = np.column_stack([linear_pred, xgb_pred])
        ensemble_pred = self.meta.predict(X_meta)

        # Convert from log scale to dollars
        price = np.exp(ensemble_pred)[0]
        lower = np.exp(ensemble_pred - margin)[0]
        upper = np.exp(ensemble_pred + margin)[0]

        return {
            'price': price,
            'confidence': confidence,
            'range': (lower, upper),
            'disagreement': disagreement[0]
        }

    def __repr__(self):
        return f"CalibratedPredictor(thresholds=[{self.high_thresh:.4f}, {self.med_thresh:.4f}])"

