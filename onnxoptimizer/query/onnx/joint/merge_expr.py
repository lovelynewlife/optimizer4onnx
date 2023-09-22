from typing import Optional, Sequence

import onnx
from onnx import ModelProto, helper, checker, GraphProto, OperatorSetIdProto

from onnxoptimizer.query.onnx.joint.model_utils import add_prefix_model

MIN_OP_SET_VERSION = 16
MAX_OP_SET_VERSION = 18

EXCLUDE_OP_SET_DOMAIN = ["ai.onnx.ml"]


def merge_project_graphs(
        g1: GraphProto,
        g2: GraphProto,
        name: Optional[str] = None,
        doc_string: Optional[str] = None,
) -> GraphProto:
    if type(g1) is not GraphProto:
        raise ValueError("g1 argument is not an ONNX graph")
    if type(g2) is not GraphProto:
        raise ValueError("g2 argument is not an ONNX graph")

    g = GraphProto()

    g.node.extend(g1.node)
    g.node.extend(g2.node)

    input_names = set()
    for e in g1.input:
        if e.name not in input_names:
            g.input.append(e)
            input_names.add(e.name)
    for e in g2.input:
        if e.name not in input_names:
            g.input.append(e)
            input_names.add(e.name)

    g.output.extend(g1.output)
    g.output.extend(g2.output)

    g.initializer.extend(g1.initializer)
    g.initializer.extend(g2.initializer)

    g.sparse_initializer.extend(g1.sparse_initializer)
    g.sparse_initializer.extend(g2.sparse_initializer)

    g.value_info.extend(g1.value_info)
    g.value_info.extend(g2.value_info)

    g.name = name if name is not None else "_".join([g1.name, g2.name])

    if doc_string is None:
        doc_string = (
                f"Graph combining {g1.name} and {g2.name}\n"
                + g1.name
                + "\n"
                + g1.doc_string
                + "\n"
                + g2.name
                + "\n"
                + g2.doc_string
        )
    g.doc_string = doc_string

    return g


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

    opset_import_map = {}
    opset_imports = list(m1.opset_import) + list(m2.opset_import)

    for entry in opset_imports:
        if entry.domain in opset_import_map:
            found_entry = opset_import_map[entry.domain]
            found_version = found_entry.version
            if entry.version != found_version:
                #               raise ValueError(
                #                   "Can't merge two models with different operator set ids for a given domain. "
                #                   f"Got: {m1.opset_import} and {m2.opset_import}"
                #               )
                opset_import_map[entry.domain] = found_entry if found_entry.version < entry.version else entry
        else:
            opset_import_map[entry.domain] = entry

            if entry.domain not in EXCLUDE_OP_SET_DOMAIN:
                if (opset_import_map[entry.domain].version > MAX_OP_SET_VERSION or
                        opset_import_map[entry.domain].version < MIN_OP_SET_VERSION):
                    opset_import_map[entry.domain].version = MAX_OP_SET_VERSION

    opset_imports: Optional[Sequence[OperatorSetIdProto]] = list(opset_import_map.values())

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


def merge_project_models_wrap(
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
    if m1.ir_version != m2.ir_version:
        ir_version = min(m1.ir_version, m2.ir_version)
        m1.ir_version = ir_version
        m2.ir_version = ir_version

    return merge_project_models(
        m1=m1,
        m2=m2,
        prefix1=prefix1,
        prefix2=prefix2,
        name=name,
        doc_string=doc_string,
        producer_name=producer_name,
        producer_version=producer_version,
        domain=domain,
        model_version=model_version
    )


def merge_graphs_binary(
        g1: GraphProto,
        g2: GraphProto,
        gop: GraphProto,
        name: Optional[str] = None,
        doc_string: Optional[str] = None,
) -> GraphProto:
    if type(g1) is not GraphProto:
        raise ValueError("g1 argument is not an ONNX graph")
    if type(g2) is not GraphProto:
        raise ValueError("g2 argument is not an ONNX graph")
    if type(gop) is not GraphProto:
        raise ValueError("gop argument is not an ONNX graph")

    g = GraphProto()

    g.node.extend(g1.node)
    g.node.extend(g2.node)
    g.node.extend(gop.node)

    input_names = set()
    for e in g1.input:
        if e.name not in input_names:
            g.input.append(e)
            input_names.add(e.name)
    for e in g2.input:
        if e.name not in input_names:
            g.input.append(e)
            input_names.add(e.name)

    g.output.extend(gop.output)

    g.initializer.extend(g1.initializer)
    g.initializer.extend(g2.initializer)
    g.initializer.extend(gop.initializer)

    g.sparse_initializer.extend(g1.sparse_initializer)
    g.sparse_initializer.extend(g2.sparse_initializer)
    g.sparse_initializer.extend(gop.sparse_initializer)

    g.value_info.extend(g1.value_info)
    g.value_info.extend(g2.value_info)
    g.value_info.extend(gop.value_info)

    g.name = name if name is not None else "_".join([g1.name, g2.name, gop.name])

    if doc_string is None:
        doc_string = (
                f"Graph combining {g1.name} and {g2.name}\n"
                + g1.name
                + "\n"
                + g1.doc_string
                + "\n"
                + g2.name
                + "\n"
                + g2.doc_string
        )
    g.doc_string = doc_string

    return g


def merge_models_binary(
        m1: ModelProto,
        m2: ModelProto,
        mop: ModelProto,
        prefix1: Optional[str] = None,
        prefix2: Optional[str] = None,
        name: Optional[str] = None,
        doc_string: Optional[str] = None,
        producer_name: Optional[str] = "onnx.compose.merge_models",
        producer_version: Optional[str] = "1.0",
        domain: Optional[str] = "",
        model_version: Optional[int] = 1,
):
    if type(m1) is not ModelProto:
        raise ValueError("m1 argument is not an ONNX model")
    if type(m2) is not ModelProto:
        raise ValueError("m2 argument is not an ONNX model")
    if type(mop) is not ModelProto:
        raise ValueError("mop argument is not an ONNX model")

    if m1.ir_version == m2.ir_version == mop.ir_version:
        ir_version = m1.ir_version
    else:
        raise ValueError(
            f"IR version mismatch {m1.ir_version} != {m2.ir_version}."
            " Both models should have the same IR version"
        )

    opset_import_map = {}
    opset_imports = list(m1.opset_import) + list(m2.opset_import) + list(mop.opset_import)

    for entry in opset_imports:
        if entry.domain in opset_import_map:
            found_entry = opset_import_map[entry.domain]
            found_version = found_entry.version
            if entry.version != found_version:
                #               raise ValueError(
                #                   "Can't merge two models with different operator set ids for a given domain. "
                #                   f"Got: {m1.opset_import} and {m2.opset_import}"
                #               )
                opset_import_map[entry.domain] = found_entry if found_entry.version < entry.version else entry
        else:
            opset_import_map[entry.domain] = entry

        if entry.domain not in EXCLUDE_OP_SET_DOMAIN:
            if (opset_import_map[entry.domain].version > MAX_OP_SET_VERSION or
                    opset_import_map[entry.domain].version < MIN_OP_SET_VERSION):
                opset_import_map[entry.domain].version = MAX_OP_SET_VERSION

    opset_imports: Optional[Sequence[OperatorSetIdProto]] = list(opset_import_map.values())

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

    graph = merge_graphs_binary(
        m1.graph,
        m2.graph,
        mop.graph,
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

    for meta_entry in mop.metadata_props:
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
        {f.name for f in m1.functions} & {f.name for f in m2.functions} &
        {f.name for f in mop.functions}
    )
    if function_overlap:
        raise ValueError(
            "Can't merge models with overlapping local function names."
            " Found in both graphs: " + ", ".join(function_overlap)
        )
    model.functions.MergeFrom(m1.functions)
    model.functions.MergeFrom(m2.functions)
    model.functions.MergeFrom(mop.functions)

    # walk around bug from sklearn-onnx converter
    # zipmap will not pass the check
    # see: https://github.com/onnx/sklearn-onnx/issues/858
    checker.check_model(model, full_check=False)

    return model


def merge_models_binary_wrap(
        m1: ModelProto,
        m2: ModelProto,
        mop: ModelProto,
        prefix1: Optional[str] = None,
        prefix2: Optional[str] = None,
        name: Optional[str] = None,
        doc_string: Optional[str] = None,
        producer_name: Optional[str] = "onnx.compose.merge_models",
        producer_version: Optional[str] = "1.0",
        domain: Optional[str] = "",
        model_version: Optional[int] = 1,
):
    if not (m1.ir_version == m2.ir_version == mop.ir_version):
        ir_version = min(m1.ir_version, m2.ir_version, mop.ir_version)
        m1.ir_version = ir_version
        m2.ir_version = ir_version
        mop.ir_version = ir_version

    return merge_models_binary(
        m1=m1,
        m2=m2,
        mop=mop,
        prefix1=prefix1,
        prefix2=prefix2,
        name=name,
        doc_string=doc_string,
        producer_name=producer_name,
        producer_version=producer_version,
        domain=domain,
        model_version=model_version
    )


def merge_graphs_unary(
        g1: GraphProto,
        gop: GraphProto,
        name: Optional[str] = None,
        doc_string: Optional[str] = None,
) -> GraphProto:
    if type(g1) is not GraphProto:
        raise ValueError("g1 argument is not an ONNX graph")
    if type(gop) is not GraphProto:
        raise ValueError("gop argument is not an ONNX graph")

    g = GraphProto()

    g.node.extend(g1.node)
    g.node.extend(gop.node)

    input_names = set()
    for e in g1.input:
        if e.name not in input_names:
            g.input.append(e)
            input_names.add(e.name)

    g.output.extend(gop.output)

    g.initializer.extend(g1.initializer)
    g.initializer.extend(gop.initializer)

    g.sparse_initializer.extend(g1.sparse_initializer)
    g.sparse_initializer.extend(gop.sparse_initializer)

    g.value_info.extend(g1.value_info)
    g.value_info.extend(gop.value_info)

    g.name = name if name is not None else "_".join([g1.name, gop.name])

    if doc_string is None:
        doc_string = (
                f"Graph unary combining {g1.name}"
                + g1.name
                + "\n"
                + g1.doc_string
                + "\n"
        )
    g.doc_string = doc_string

    return g


def merge_models_unary(
        m1: ModelProto,
        mop: ModelProto,
        prefix1: Optional[str] = None,
        name: Optional[str] = None,
        doc_string: Optional[str] = None,
        producer_name: Optional[str] = "onnx.compose.merge_models",
        producer_version: Optional[str] = "1.0",
        domain: Optional[str] = "",
        model_version: Optional[int] = 1,
):
    if type(m1) is not ModelProto:
        raise ValueError("m1 argument is not an ONNX model")
    if type(mop) is not ModelProto:
        raise ValueError("mop argument is not an ONNX model")

    if m1.ir_version == mop.ir_version:
        ir_version = m1.ir_version
    else:
        raise ValueError(
            f"IR version mismatch {m1.ir_version} != {mop.ir_version}."
            " Both models should have the same IR version"
        )

    opset_import_map = {}
    opset_imports = list(m1.opset_import) + list(mop.opset_import)

    for entry in opset_imports:
        if entry.domain in opset_import_map:
            found_entry = opset_import_map[entry.domain]
            found_version = found_entry.version
            if entry.version != found_version:
                #               raise ValueError(
                #                   "Can't merge two models with different operator set ids for a given domain. "
                #                   f"Got: {m1.opset_import} and {m2.opset_import}"
                #               )
                opset_import_map[entry.domain] = found_entry if found_entry.version < entry.version else entry
        else:
            opset_import_map[entry.domain] = entry

        if entry.domain not in EXCLUDE_OP_SET_DOMAIN:
            if (opset_import_map[entry.domain].version > MAX_OP_SET_VERSION or
                    opset_import_map[entry.domain].version < MIN_OP_SET_VERSION):
                opset_import_map[entry.domain].version = MAX_OP_SET_VERSION

    opset_imports: Optional[Sequence[OperatorSetIdProto]] = list(opset_import_map.values())

    # Prefixing names in the graph if requested, adjusting io_map accordingly
    if prefix1:
        m1_copy = ModelProto()
        m1_copy.CopyFrom(m1)
        m1 = m1_copy
        m1 = add_prefix_model(m1, prefix=prefix1)

    graph = merge_graphs_unary(
        m1.graph,
        mop.graph,
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

    for meta_entry in mop.metadata_props:
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
        {f.name for f in m1.functions} &
        {f.name for f in mop.functions}
    )
    if function_overlap:
        raise ValueError(
            "Can't merge models with overlapping local function names."
            " Found in both graphs: " + ", ".join(function_overlap)
        )
    model.functions.MergeFrom(m1.functions)
    model.functions.MergeFrom(mop.functions)

    # walk around bug from sklearn-onnx converter
    # zipmap will not pass the check
    # see: https://github.com/onnx/sklearn-onnx/issues/858
    checker.check_model(model, full_check=False)

    return model


def merge_models_unary_wrap(
        m1: ModelProto,
        mop: ModelProto,
        prefix1: Optional[str] = None,
        name: Optional[str] = None,
        doc_string: Optional[str] = None,
        producer_name: Optional[str] = "onnx.compose.merge_models",
        producer_version: Optional[str] = "1.0",
        domain: Optional[str] = "",
        model_version: Optional[int] = 1,
):
    if not (m1.ir_version == mop.ir_version):
        ir_version = min(m1.ir_version, mop.ir_version)
        m1.ir_version = ir_version
        mop.ir_version = ir_version

    return merge_models_unary(
        m1=m1,
        mop=mop,
        prefix1=prefix1,
        name=name,
        doc_string=doc_string,
        producer_name=producer_name,
        producer_version=producer_version,
        domain=domain,
        model_version=model_version
    )
