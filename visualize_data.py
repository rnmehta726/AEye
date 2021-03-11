import os
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

# initialize COCO API for instance annotations
dataDir = 'cocoapi'
dataType = 'val2017'
instances_annFile =r'cocoapi\PythonAPI\annotations\instances_val2017.json'
coco = COCO(instances_annFile)

# initialize COCO API for caption annotations
captions_annFile = r'cocoapi\PythonAPI\annotations\captions_val2017.json'
coco_caps = COCO(captions_annFile)

# get image ids 
ids = list(coco.anns.keys())

# pick a random image and obtain the corresponding URL
ann_id = np.random.choice(ids)
img_id = coco.anns[ann_id]['image_id']
img = coco.loadImgs(img_id)[0]
url = img['coco_url']

# print URL and visualize corresponding image
print(url)
I = io.imread(url)
plt.axis('off')
plt.imshow(I)
plt.show()

# load and display captions
annIds = coco_caps.getAnnIds(imgIds=img['id'])
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)