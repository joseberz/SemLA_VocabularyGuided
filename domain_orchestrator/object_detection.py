import json

from ultralytics import YOLOWorld

# TODO evtl. anders an die Vokabulare der Domänen kommen als hier hardcoded
ADAPTER_VOCAB_JSONS = {
    "mv":              "../catseg/datasets/mv65.json",
    "cs-normal":       "../catseg/datasets/cs19.json",
    "pc59":            "../catseg/datasets/pc59.json",
    "a133":            "../catseg/datasets/ade133.json",
    "coconutL":        "../catseg/datasets/coconut.json",
    "nyu":             "../catseg/datasets/nyu40.json",
    "idd":             "../catseg/datasets/idd30.json",
    "iddnovel":        "../catseg/datasets/idd30.json",
    "nyunovel":        "../catseg/datasets/nyu40.json",
    "pc59novel":       "../catseg/datasets/pc59.json",
    "acdc-night":      "../catseg/datasets/cs19.json",
    "acdc-rain":       "../catseg/datasets/cs19.json",
    "acdc-snow":       "../catseg/datasets/cs19.json",
    "acdc-fog":        "../catseg/datasets/cs19.json",
    "muses-clear-day": "../catseg/datasets/cs19.json",
    "muses-clear-night": "../catseg/datasets/cs19.json",
    "muses-rain-day": "../catseg/datasets/cs19.json",
    "muses-rain-night": "../catseg/datasets/cs19.json",
    "muses-fog-day": "../catseg/datasets/cs19.json",
    "muses-fog-night": "../catseg/datasets/cs19.json",
    "muses-snow-day": "../catseg/datasets/cs19.json",
    "muses-snow-night": "../catseg/datasets/cs19.json",
    "bdd": "../catseg/datasets/cs19.json"
}

class ObjectDetector:
    """Handles image and dataset embedding operations."""

    def __init__(self):
        self.model = YOLOWorld("yolov8m-worldv2.pt")
        self.classes = []

        for adapter_name, json_path in ADAPTER_VOCAB_JSONS.items():
            with open(json_path) as f:
                class_names = json.load(f)
                class_names = [class_name.lower() for class_name in class_names]
                self.classes.extend(class_names)

        self.classes = list(set(self.classes))
        self.model.set_classes(self.classes)

    def detect_objects(self, image_path):
        return self.model.predict(image_path, conf=0.15, verbose=False)