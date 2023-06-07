import os

DATA_PATH = os.getenv("DATA_PATH")

TIME_SERIES_LENGTH = 70
FOLDS = None

NUM_CLASSES = 20
LABEL_NAMES = {
  "0": "Background",
  "1": "Meadow",
  "2": "Soft winter wheat",
  "3": "Corn",
  "4": "Winter barley",
  "5": "Winter rapeseed",
  "6": "Spring barley",
  "7": "Sunflower",
  "8": "Grapevine",
  "9": "Beet",
  "10": "Winter triticale",
  "11": "Winter durum wheat",
  "12": "Fruits,  vegetables, flowers",
  "13": "Potatoes",
  "14": "Leguminous fodder",
  "15": "Soybeans",
  "16": "Orchard",
  "17": "Mixed cereal",
  "18": "Sorghum",
  "19": "Void label"
}
