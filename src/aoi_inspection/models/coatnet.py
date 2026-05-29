from __future__ import annotations

from coatnet_pytorch_master.coatnet import (
    coatnet_0,
    coatnet_1,
    coatnet_2,
    coatnet_3,
    coatnet_4,
)


MODEL_FACTORY = {
    "coatnet_0": coatnet_0,
    "coatnet_1": coatnet_1,
    "coatnet_2": coatnet_2,
    "coatnet_3": coatnet_3,
    "coatnet_4": coatnet_4,
}


def build_model(name: str, num_classes: int):
    try:
        factory = MODEL_FACTORY[name]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_FACTORY))
        raise ValueError(f"Unknown model '{name}'. Available models: {choices}") from exc
    return factory(num_classes=num_classes)
