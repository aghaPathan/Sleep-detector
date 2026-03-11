"""Download the dlib shape predictor model required by the sleep detector."""

import bz2
import os
import sys
import urllib.request

MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
MODEL_FILE = "shape_predictor_68_face_landmarks.dat"
COMPRESSED_FILE = MODEL_FILE + ".bz2"


def download_model(dest_dir=None):
    """Download and extract the dlib 68-point face landmark model.

    Args:
        dest_dir: Directory to save the model in. Defaults to script directory.
    """
    if dest_dir is None:
        dest_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(dest_dir, MODEL_FILE)
    compressed_path = os.path.join(dest_dir, COMPRESSED_FILE)

    if os.path.isfile(model_path):
        print(f"Model already exists: {model_path}")
        return model_path

    print(f"Downloading {MODEL_URL} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, compressed_path)
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("Extracting...")
    with bz2.open(compressed_path, "rb") as src, open(model_path, "wb") as dst:
        dst.write(src.read())

    os.remove(compressed_path)
    print(f"Model saved to: {model_path}")
    return model_path


if __name__ == "__main__":
    download_model()
