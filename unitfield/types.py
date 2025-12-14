from typing import Tuple, List
from typing import Annotated
from numpy.typing import NDArray
from boundednumbers import UnitFloat
import numpy as np

UnitArray = Annotated[NDArray[np.floating], "values in [0, 1]"]
VectorReturnType = Tuple[UnitFloat, ...] | List[UnitFloat] | UnitArray