"""Optional model regression; run explicitly because it loads YOLO-World."""
import unittest
from pathlib import Path

from ultralytics import YOLO


class WrapperDetectionRegression(unittest.TestCase):
    def test_reference_wrapper_is_candidate(self):
        image = Path("tests/fixtures/wrapper.png")
        if not image.exists():
            self.skipTest("reference fixture not installed")
        model = YOLO("yolov8s-worldv2.pt")
        model.set_classes(["discarded packaging", "piece of litter", "plastic wrapper"])
        result = model.predict(image, conf=.004, imgsz=640, verbose=False)[0]
        self.assertTrue(any(result.names[int(box.cls.item())] == "discarded packaging" for box in result.boxes))


if __name__ == "__main__":
    unittest.main()
