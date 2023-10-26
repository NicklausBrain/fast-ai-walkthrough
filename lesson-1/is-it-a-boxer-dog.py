# pip install -U duckduckgo_search
# pip install -U fastcore
# pip install -U fastai

from duckduckgo_search import DDGS
from fastdownload import download_url
from fastcore.all import *
from fastai.vision.all import *

def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(DDGS().images(keywords=term, max_results=max_images)).itemgot('image')

searches = 'wild animal','boxer dog'
path = Path('boxer-dog_or_not')
from time import sleep

for o in searches:
    dest = (path/o)
    if(dest.exists()):
        break
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    sleep(10)  # Pause between searches to avoid over-loading server
    download_images(dest, urls=search_images(f'{o} sun photo'))
    sleep(10)
    download_images(dest, urls=search_images(f'{o} shade photo'))
    sleep(10)
    resize_images(path/o, max_size=400, dest=path/o)

# Some photos might not download correctly which could cause our model training to fail, so we'll remove them:
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)


# To train a model, we'll need DataLoaders,
#  which is an object that contains a training set (the images used to create a model)
#  and a validation set (the images used to check the accuracy of a model -- not used during training). 
# In fastai we can create that easily using a DataBlock, and view sample images from it:

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

dls.show_batch(max_n=6)

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)

### Positive test
dest = Path('boxer-dog.jpg')
if(not dest.exists()):
    urls = search_images('boxer puppy going crazy', max_images=1)
    download_url(urls[0], dest, show_progress=True)

im = Image.open(dest)
im.to_thumb(256,256)
is_boxer,_,probs = learn.predict(PILImage.create(dest))
print(f"This is a: {is_boxer}.")
print(f"Probability it's a boxer: {probs[0]:.4f}")

### Negative test
destX = Path('chihuahua.jpg')
if(not destX.exists()):
    urlsX = search_images('chihuahua', max_images=10)
    download_url(urlsX[1], destX, show_progress=False)

imX = Image.open(destX)
imX.to_thumb(256,256)

is_boxerX,_,probsX = learn.predict(PILImage.create(destX))
print(f"This is a: {is_boxerX}.")
print(f"Probability it's a boxer: {probsX[0]:.4f}")
