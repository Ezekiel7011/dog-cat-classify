from aoi_inspection.config import parse_config


def test_parse_config_validates_class_count():
    config = parse_config(
        {
            "project": {"seed": 7},
            "data": {"classes": ["pass", "defect"], "image_size": [224, 224]},
            "model": {"name": "coatnet_0", "num_classes": 2},
            "artifacts": {},
        }
    )

    assert config.seed == 7
    assert config.data.classes == ["pass", "defect"]
    assert config.data.image_size == (224, 224)
