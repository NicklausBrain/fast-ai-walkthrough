# pip install -U duckduckgo_search
# pip install -U fastcore
# pip install -U fastai

from fastcore.all import *
from fastai.vision.all import *
import sys

if len(sys.argv) <= 1:
    print("No arguments were given")
    exit(-1)

# read https://benjaminwarner.dev/2021/10/01/inference-with-fastai
learn = load_learner('vehicle-types.pkl', cpu=False)

image_path = Path(sys.argv[1]);
if not image_path.exists():
    print("Given path does not exist")
    exit(-1)

vehicle_type,_,probs = learn.predict(PILImage.create(image_path))
print(f"This is a: {vehicle_type}.")
print(f"Probability: {probs[0]:.4f}")
print(probs)