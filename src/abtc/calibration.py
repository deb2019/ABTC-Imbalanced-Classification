from typing import Tuple
import numpy as np
from sklearn.isotonic import IsotonicRegression

def isotonic_calibrator(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[IsotonicRegression, np.ndarray]:
    ir = IsotonicRegression(out_of_bounds="clip")
    y_cal = ir.fit_transform(y_score, y_true)
    return ir, np.asarray(y_cal)
