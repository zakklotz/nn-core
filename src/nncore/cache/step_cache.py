class StepCache:
    """
    Generic per-layer cache for step-wise (e.g., recursion depth) reuse.
    Stores arbitrary objects keyed by (layer_idx, key).
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self._store: list[dict[str, object]] = [dict() for _ in range(num_layers)]

    def _validate_layer(self, layer_idx: int) -> None:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range for num_layers={self.num_layers}.")

    def reset(self) -> None:
        for layer in self._store:
            layer.clear()

    def get(self, layer_idx: int, key: str) -> object | None:
        self._validate_layer(layer_idx)
        return self._store[layer_idx].get(key)

    def set(self, layer_idx: int, key: str, value: object) -> None:
        self._validate_layer(layer_idx)
        self._store[layer_idx][key] = value
