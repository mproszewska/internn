def squeeze_2D(array, op="max"):
    if len(array.shape) == 2:
        return array
    if op == "max":
        return array.max(axis=2)
    if op == "mean":
        return array.mean(axis=2)
    raise ValueError("Invalid map_op value: {}".format(op))
    

def scale_to_image(array):
    array_scaled = (array - array.min()) / (array.max() - array.min())
    image = (255.0 * array_scaled).astype("uint8")
    return image
    