import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from services.classifier import detect_waste, get_model_path


class WasteClassifierTests(unittest.TestCase):
    def test_maps_model_classes_and_keeps_confidence_and_box(self):
        box = SimpleNamespace(
            cls=SimpleNamespace(item=lambda: 5),
            conf=SimpleNamespace(item=lambda: 0.87),
            xyxy=[SimpleNamespace(tolist=lambda: [10.0, 20.0, 50.0, 80.0])],
        )
        prediction = SimpleNamespace(names={5: "Clear plastic bottle"}, boxes=[box])

        class Model:
            def predict(self, image, conf, verbose):
                self.assertion = (image.size, conf, verbose)
                return [prediction]

        model = Model()
        result = detect_waste(model, Image.new("RGB", (100, 100)), 0.4)

        self.assertEqual(result, [{
            "category": "Plastic",
            "confidence": 0.87,
            "box": [10.0, 20.0, 50.0, 80.0],
            "class_name": "Clear plastic bottle",
        }])
        self.assertEqual(model.assertion, ((100, 100), 0.4, False))

    def test_invalid_explicit_model_path_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_model = str(Path(directory) / "missing.pt")
            with patch.dict(os.environ, {"GARBAGE_MODEL_PATH": missing_model}):
                with self.assertRaisesRegex(FileNotFoundError, "GARBAGE_MODEL_PATH"):
                    get_model_path()

    def test_invalid_confidence_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            detect_waste(object(), Image.new("RGB", (2, 2)), 1.1)


if __name__ == "__main__":
    unittest.main()
