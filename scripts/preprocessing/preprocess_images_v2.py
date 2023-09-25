import argparse
from PIL import Image 
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import sys

def format_xml(root, r):
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')

        xmin = int(bbox.find('xmin').text)
        xmax = int(bbox.find('xmax').text)
        ymin = int(bbox.find('ymin').text)
        ymax = int(bbox.find('ymax').text)

        xmin = xmin / r
        xmax = xmax / r
        ymin = ymin / r
        ymax = ymax / r

        bbox.find('xmin').text = str(int(xmin))
        bbox.find('xmax').text = str(int(xmax))
        bbox.find('ymin').text = str(int(ymin))
        bbox.find('ymax').text = str(int(ymax))

    return root        

def format_image(img, input_size):
    img_arr = np.array(img)
    height, width, _ = img_arr.shape
    max_size = max(height, width)
    r = max_size / input_size
    new_width = int(width / r)
    new_height = int(height / r)
    new_size = (new_width, new_height)

    resized = cv2.resize(img_arr, new_size, interpolation= cv2.INTER_LINEAR)
    new_image = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    new_image[0:new_height, 0:new_width, :] = resized
    new_image = Image.fromarray(new_image)
    return new_image, r

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_size", help="Image new size")
    parser.add_argument("source", help="Image source folder")
    parser.add_argument("destination", help="Image destination folder")
    parser.add_argument("name", help="Image name")

    args = parser.parse_args()

    index = 0
    for img in os.listdir(args.source):
        if os.path.isfile(os.path.join(args.source, img)) and ".xml" not in img:
            print(img)
            tree = ET.parse(os.path.join(args.source, img[:-4] + ".xml"))
            root = tree.getroot()

            img = Image.open(os.path.join(args.source, img))
            img = img.convert('RGB')
            
            new_img, r = format_image(img, int(args.input_size))
            new_img_name = args.name + str(index)
            root = format_xml(root, r)

            new_img.save(os.path.join(args.destination, new_img_name + '.jpg'))
            tree.write(os.path.join(args.destination, new_img_name + ".xml"))

            index += 1