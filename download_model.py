import os
import gdown

os.makedirs("models", exist_ok=True)

MODEL_PATH = "models/model.h5"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(
        id="10nGYGyXCUJI8Nbz3hPpRDb9H26lQeG8A",
        output=MODEL_PATH,
        quiet=False,
    )
else:
    print("Model already exists.")
