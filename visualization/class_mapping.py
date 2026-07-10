"""
TODO DOKU
"""


# Normierung der Klassennamen über Domänengrenzen hinweg
CANONICAL_CLASS_MAP: dict[str, str] = {

    "motorbike": "motorcycle",
    "minibike": "motorcycle",
    "on rails": "train",
    "aeroplane": "airplane",
    "rail track": "railroad",
    "traffic sign (front)": "traffic sign",
    "traffic sign (back)": "traffic sign",

    "bicyclist": "rider",
    "motorcyclist": "rider",
    "other rider": "rider",

    "ground animal": "animal",

    "couch": "sofa",
    "shelf": "shelves",
    "pottedplant": "potted plant",
    "diningtable": "dining table",
    "dining table": "table",
    "countertop": "counter",
    "books": "book",
    "clothes": "apparel",
    "cloth": "apparel",
    "picture": "painting",
    "ball": "sports ball",
    "bookshelf": "bookcase",
    "lamp": "light",
    "stove": "oven",

    "tvmonitor": "television",
    "tv": "television",
    "television receiver": "television",

    "windowpane": "window",
    "blind": "blinds",
    "window blind": "blinds",

    "streetlight": "street light",
    "utility pole": "pole",

    "ashcan": "trash can",

    "brick wall": "wall",
    "stone wall": "wall",
    "tile wall": "wall",
    "wood wall": "wall",
    "house": "building",
    "skyscraper": "building",
    "pavement": "sidewalk",
    "wood floor": "floor",

    "orange fruit": "orange",
    "tree": "vegetation",

    "earth": "terrain",
    "land": "terrain",
    "dirt": "terrain",
    "gravel": "terrain",
    "ground": "terrain",

    "backpack": "bag",
    "handbag": "bag",
}
