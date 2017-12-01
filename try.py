import numpy as np
from imutils import contours
import imutils
import argparse
import pytesseract as tesseract
from PIL import Image

text=tesseract.image_to_string(Image.open('detailed_ROI.tiff'))
print (text)
