# pip install -U duckduckgo_search
# pip install -U fastcore
# pip install -U fastai

from duckduckgo_search import DDGS
from fastdownload import download_url
from fastcore.all import *
from fastai.vision.all import *

from duckduckgo_search import DDGS
from fastdownload import download_url

cat_url = L(DDGS().images(keywords='cat', max_results=1)).itemgot('image')[0]

# print (cat_url)

img = PILImage.create(download_url(cat_url))
img.to_thumb(192)