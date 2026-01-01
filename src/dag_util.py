import tensorflow as tf

def get_previous(model, name):
    inbound = model.get_layer(name).inbound_nodes[0].inbound_layers
    if type(inbound) != list:
        inbound = [inbound]
    return [layer.name for layer in inbound]


def traverse_improved(model, name, start, part_name, inpt, tensor_cache):
    # base case：reach the start layer
    if name == start:
        return model.get_layer(start).output  # use the original model's output at start
    if name == part_name:
        return inpt

    # check cache
    if name in tensor_cache:
        return tensor_cache[name]

    # recursively get inputs from previous layers
    input_tensors = []
    for n in get_previous(model, name):
        # recursive call
        input_tensors.append(traverse_improved(model, n, start, part_name, inpt, tensor_cache))

    # prepare input for current layer
    if len(input_tensors) == 1:
        # single input, unwrap from list
        current_input = input_tensors[0]
    else:
        # multiple inputs, keep as list
        current_input = input_tensors

    # call the current layer
    layer = model.get_layer(name)

    try:
        # call layer with current_input
        to_next = layer(current_input)
    except Exception as e:
        raise RuntimeError(
            f"Error calling layer {name} with inputs: {current_input}. The original model's connection might involve a merge layer that needs to be explicitly handled.") from e

    # store in cache
    tensor_cache[name] = to_next

    return to_next

def construct_model(model, start, end, part_name="part_begin"):
    # 1. create Input tensor for the start layer's output
    input_tensor = tf.keras.Input(tensor=model.get_layer(start).output, name=part_name)

    # tensor cache to store intermediate tensors
    tensor_cache = {part_name: input_tensor}

    # 2. traverse the model from end to start
    output_tensor = traverse_improved(model, end, start, part_name, input_tensor, tensor_cache)

    # 3. create new model
    part = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    return part
