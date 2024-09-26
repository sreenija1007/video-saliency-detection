# from io import BytesIO
# import os
# import base64
# from skimage import io, transform, color
# # data loader
# import glob
# import torch
# from skimage import io, transform, color
# import numpy as np
# import random
# import math
# import imageio
# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
from PIL import Image
# from pathlib import Path

# path = os.path.normpath("C:\\Users\\13013\\Desktop\\removebg\\static\\uploads\\test.JPG")

# wcubic = Image.open(path)
# wcubic.save("converted.png", format = "png")
# converted = Image.open("converted.png")
# data = BytesIO()
# converted.save(data, "PNG")
# data64 = base64.b64encode(data.getvalue())
# print(imageio.imread("data:img/PNG;base64,"+data64.decode("ascii"), pilmode ="RGB"))


# image = io.imread(path)
# print(image)
# #print(type(image[0][0][0]))

import os
import glob
from pathlib import Path

# files = glob.glob("\\static\\uploads")
# print(files)
# for f in files:
#     os.remove(f)
#     print("did it?")



# filepath = "\\static\\uploads\\pic1.PNG"
# try:
#     os.remove(filepath)
# except OSError as e:
#     print("Error: %s:%s" % (filepath, e.strerror))

file_path = Path("\\static\\uploads\\pic1.PNG")
image_dir = os.path.join(os.getcwd(), 'static', 'uploads')
img_name_list = glob.glob(image_dir + os.sep + '*')
for img in img_name_list:
    print(img)
    image = Image.open(img)
    image.thumbnail((100,100))
    print(type(image))
