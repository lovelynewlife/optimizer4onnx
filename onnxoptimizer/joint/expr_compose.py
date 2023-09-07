from typing import MutableMapping, Optional

from onnx import ModelProto, helper, checker

from onnxoptimizer.joint.model_utils import add_prefix_model, merge_project_graphs


def merge_project_models(
        m1: ModelProto,
        m2: ModelProto,
        prefix1: Optional[str] = None,
        prefix2: Optional[str] = None,
        name: Optional[str] = None,
        doc_string: Optional[str] = None,
        producer_name: Optional[str] = "onnx.expr_compose.merge_models",
        producer_version: Optional[str] = "1.0",
        domain: Optional[str] = "",
        model_version: Optional[int] = 1, ) -> ModelProto:
    if type(m1) is not ModelProto:
        raise ValueError("m1 argument is not an ONNX model")
    if type(m2) is not ModelProto:
        raise ValueError("m2 argument is not an ONNX model")

    if m1.ir_version != m2.ir_version:
        raise ValueError(
            f"IR version mismatch {m1.ir_version} != {m2.ir_version}."
            " Both models should have the same IR version"
        )
    ir_version = m1.ir_version

    opset_import_map: MutableMapping[str, int] = {}
    opset_imports = list(m1.opset_import) + list(m2.opset_import)

    for entry in opset_imports:
        if entry.domain in opset_import_map:
            found_version = opset_import_map[entry.domain]
            if entry.version != found_version:
                raise ValueError(
                    "Can't merge two models with different operator set ids for a given domain. "
                    f"Got: {m1.opset_import} and {m2.opset_import}"
                )
        else:
            opset_import_map[entry.domain] = entry.version

    # Prefixing names in the graph if requested, adjusting io_map accordingly
    if prefix1 or prefix2:
        if prefix1:
            m1_copy = ModelProto()
            m1_copy.CopyFrom(m1)
            m1 = m1_copy
            m1 = add_prefix_model(m1, prefix=prefix1)
        if prefix2:
            m2_copy = ModelProto()
            m2_copy.CopyFrom(m2)
            m2 = m2_copy
            m2 = add_prefix_model(m2, prefix=prefix2)

    graph = merge_project_graphs(
        m1.graph,
        m2.graph,
        name=name,
        doc_string=doc_string,
    )
    model = helper.make_model(
        graph,
        producer_name=producer_name,
        producer_version=producer_version,
        domain=domain,
        model_version=model_version,
        opset_imports=opset_imports,
        ir_version=ir_version,
    )

    # Merging model metadata props
    model_props = {}
    for meta_entry in m1.metadata_props:
        model_props[meta_entry.key] = meta_entry.value
    for meta_entry in m2.metadata_props:
        if meta_entry.key in model_props:
            value = model_props[meta_entry.key]
            if value != meta_entry.value:
                raise ValueError(
                    "Can't merge models with different values for the same model metadata property."
                    f" Found: property = {meta_entry.key}, with values {value} and {meta_entry.value}."
                )
        else:
            model_props[meta_entry.key] = meta_entry.value
    helper.set_model_props(model, model_props)

    # Merging functions
    function_overlap = list(
        {f.name for f in m1.functions} & {f.name for f in m2.functions}
    )
    if function_overlap:
        raise ValueError(
            "Can't merge models with overlapping local function names."
            " Found in both graphs: " + ", ".join(function_overlap)
        )
    model.functions.MergeFrom(m1.functions)
    model.functions.MergeFrom(m2.functions)

    # walk around bug from sklearn-onnx converter
    # zipmap will not pass the check
    # see: https://github.com/onnx/sklearn-onnx/issues/858
    checker.check_model(model, full_check=False)

    return model
