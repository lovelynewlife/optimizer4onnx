import numpy as np
from onnxconverter_common import (Int64TensorType, Int32TensorType,
                                  FloatTensorType, StringTensorType)

numpy_type_map = {
    np.float32: np.float32,
    np.float64: np.float32,
    np.int64: np.int64,
    np.int32: np.int32,
    np.object_: str,
}

numpy_onnx_type_map = {
    np.int64: Int64TensorType([None, 1]),
    np.int32: Int32TensorType([None, 1]),
    np.float64: FloatTensorType([None, 1]),
    np.float32: FloatTensorType([None, 1]),
    np.object_: StringTensorType([None, 1]),
}
