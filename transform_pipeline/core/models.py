from dataclasses import dataclass
import numpy as np
@dataclass(frozen=True)
class RasterGeometry:
    rows:int; columns:int; x_axis:np.ndarray; y_axis:np.ndarray
@dataclass(frozen=True)
class ShiftMeasurement:
    row:int; raw_shift:float; smoothed_shift:float; peak_correlation:float; zero_correlation:float; correlation_gain:float
@dataclass
class ReconstructionResult:
    cube:np.ndarray; total_counts:np.ndarray; geometry:RasterGeometry
@dataclass
class RegistrationResult:
    cube:np.ndarray; total_counts:np.ndarray; validity_mask:np.ndarray; shifts:list[ShiftMeasurement]
