from PIL import Image
import math

from os import listdir
from os.path import isfile, join
files = [ f for f in listdir("./collage/") if isfile(join("./collage/", f)) ]

files.sort(key=lambda f : f.split(".")[0])
files.sort(key=lambda f : f.split(".")[1])

IM_WIDTH = 2000
IM_HEIGHT = 2800
NUM_IMAGES = len(files)
NUM_ROWS = math.ceil(math.sqrt(NUM_IMAGES))
NUM_COLS = math.ceil(math.sqrt(NUM_IMAGES))
COLLAGE_WIDTH = IM_WIDTH*NUM_ROWS
COLLAGE_HEIGHT = IM_HEIGHT*NUM_COLS
RESIZE_SCALE=1/5

print("creating image...")
print(f"{NUM_IMAGES} images, {NUM_COLS} columns, {NUM_ROWS} rows")
print(f"final image resolution: {COLLAGE_WIDTH}x{COLLAGE_HEIGHT}")
new_im = Image.new('RGB', (COLLAGE_WIDTH,COLLAGE_HEIGHT))

index = 0
for j in range(0,COLLAGE_HEIGHT,IM_HEIGHT):
  for i in range(0,COLLAGE_WIDTH,IM_WIDTH):
    im = Image.open("./collage/" + files[index])
    new_im.paste(im, (i,j))
    index += 1
    print(f"pasted image {index}/{NUM_IMAGES}")

RESIZED_WIDTH = int(COLLAGE_WIDTH*RESIZE_SCALE)
RESIZED_HEIGHT = int(COLLAGE_HEIGHT*RESIZE_SCALE)

if (RESIZE_SCALE != 0): new_im = new_im.resize((RESIZED_WIDTH, RESIZED_HEIGHT), Image.Resampling.LANCZOS)
print(f"resized to 1/{int(1/RESIZE_SCALE)} scale! new resolution: {RESIZED_WIDTH}x{RESIZED_HEIGHT}")
new_im.save("collage_thumb.png") # unused: optimize=True, params={'optimize': True}