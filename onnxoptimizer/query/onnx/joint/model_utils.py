from typing import Optional, Dict

from onnx import ModelProto, GraphProto


def add_prefix_graph(
        graph: GraphProto,
        prefix: str,
        input_names: set,
        sub_graph: Optional[bool] = False,
        inplace: Optional[bool] = False,
        name_map: Optional[Dict[str, str]] = None,
) -> GraphProto:
    if type(graph) is not GraphProto:
        raise ValueError("graph argument is not an ONNX graph")

    if not inplace:
        g = GraphProto()
        g.CopyFrom(graph)
    else:
        g = graph

    def _prefixed(pre: str, name: str) -> str:
        return pre + name if len(name) > 0 else name

    if name_map is None:
        name_map = {}

    if sub_graph:
        for n in g.node:
            for e in n.input:
                if e not in input_names:
                    name_map[e] = _prefixed(prefix, e)
            for e in n.output:
                name_map[e] = _prefixed(prefix, e)

        for entry in g.input:
            e = entry.name
            if e not in input_names:
                name_map[e] = _prefixed(prefix, e)
    else:
        input_names = set()
        for entry in g.input:
            input_names.add(entry.name)

        for n in g.node:
            for e in n.input:
                if e not in input_names:
                    name_map[e] = _prefixed(prefix, e)

            for e in n.output:
                name_map[e] = _prefixed(prefix, e)

    for entry in g.output:
        name_map[entry.name] = _prefixed(prefix, entry.name)

    for n in g.node:
        n.name = _prefixed(prefix, n.name)
        for attribute in n.attribute:
            if attribute.g:
                add_prefix_graph(
                    attribute.g, prefix, input_names, inplace=True, sub_graph=True, name_map=name_map
                )

    for init in g.initializer:
        name_map[init.name] = _prefixed(prefix, init.name)
    for sparse_init in g.sparse_initializer:
        name_map[sparse_init.values.name] = _prefixed(
            prefix, sparse_init.values.name
        )
        name_map[sparse_init.indices.name] = _prefixed(
            prefix, sparse_init.indices.name
        )

    for entry in g.value_info:
        name_map[entry.name] = _prefixed(prefix, entry.name)

    for n in g.node:
        for i, output in enumerate(n.output):
            if n.output[i] in name_map:
                n.output[i] = name_map[output]
        for i, input_ in enumerate(n.input):
            if n.input[i] in name_map:
                n.input[i] = name_map[input_]

    for in_desc in g.input:
        if in_desc.name in name_map:
            in_desc.name = name_map[in_desc.name]
    for out_desc in g.output:
        if out_desc.name in name_map:
            out_desc.name = name_map[out_desc.name]

    for initializer in g.initializer:
        if initializer.name in name_map:
            initializer.name = name_map[initializer.name]
    for sparse_initializer in g.sparse_initializer:
        if sparse_initializer.values.name in name_map:
            sparse_initializer.values.name = name_map[sparse_initializer.values.name]
        if sparse_initializer.indices.name in name_map:
            sparse_initializer.indices.name = name_map[sparse_initializer.indices.name]

    for value_info in g.value_info:
        if value_info.name in name_map:
            value_info.name = name_map[value_info.name]

    return g


def add_prefix_model(
        model: ModelProto,
        prefix: str,
        inplace: Optional[bool] = False,
) -> ModelProto:
    if type(model) is not ModelProto:
        raise ValueError("model argument is not an ONNX model")

    if not inplace:
        m = ModelProto()
        m.CopyFrom(model)
        model = m

    add_prefix_graph(
        model.graph,
        prefix,
        set(),
        inplace=True,  # No need to create a copy, since it's a new model
    )

    f_name_map = {}
    for f in model.functions:
        new_f_name = prefix + f.name
        f_name_map[f.name] = new_f_name
        f.name = new_f_name
    # Adjust references to local functions in other local function
    # definitions
    for f in model.functions:
        for n in f.node:
            if n.op_type in f_name_map:
                n.op_type = f_name_map[n.op_type]
    # Adjust references to local functions in the graph
    for n in model.graph.node:
        if n.op_type in f_name_map:
            n.op_type = f_name_map[n.op_type]

    return model
