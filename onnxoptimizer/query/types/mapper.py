import numpy as np
from onnx import TensorProto
from onnxconverter_common import (Int64TensorType, Int32TensorType,
                                  FloatTensorType, StringTensorType, BooleanTensorType)

numpy_type_map = {
    np.float32: np.float32,
    np.float64: np.float32,
    np.int64: np.int64,
    np.int32: np.int32,
    np.object_: str,
}

input_numpy_onnx_type_map = {
    np.int64: Int64TensorType([None, 1]),
    np.int32: Int32TensorType([None, 1]),
    np.float64: FloatTensorType([None, 1]),
    np.float32: FloatTensorType([None, 1]),
    np.object_: StringTensorType([None, 1]),
    np.bool_: BooleanTensorType([None, 1])
}

numpy_onnx_tensor_type_map = {
    np.int64: TensorProto.INT64,
    np.int32: TensorProto.INT32,
    np.float64: TensorProto.FLOAT,
    np.float32: TensorProto.FLOAT,
    np.object_: TensorProto.STRING,
    np.bool_: TensorProto.BOOL,
}

onnx_type_str_numpy_map = {
    'tensor(int64)': np.dtype("int64"),
    'tensor(float)': np.dtype("float")
}
