import os
import gdown

os.makedirs("models", exist_ok=True)

MODEL_PATH = "models/model.h5"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = "https://drive.google.com/uc?id=10nGYGyXCUJI8Nbz3hPpRDb9H26lQeG8A"
    gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
else:
    print("Model already exists.")
