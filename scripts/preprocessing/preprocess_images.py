import argparse
from PIL import Image 
import os
import sys
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()

parser.add_argument("source", help="Image source folder")
parser.add_argument("destination", help="Image destination folder")
parser.add_argument("new_width", help="Image width")
parser.add_argument("new_height", help="Image height")
parser.add_argument("extension", help="Image extension")
parser.add_argument("name", help="Image name")
parser.add_argument("-i", "--index", help="Starting index for naming")
parser.add_argument("-x", "--xml", help="Convert the xml files if exist")

args = parser.parse_args()

if not os.path.exists(args.source):
    print("Source path doesn't exist.")
    sys.exit(1)

if not os.path.isdir(args.source):
    print("Source path is not a directory.")
    sys.exit(1)

if not os.path.exists(args.destination):
    print("Destination path doesn't exist.")
    sys.exit(1)

if not os.path.isdir(args.destination):
    print("Destination path is not a directory.")
    sys.exit(1)

try:
    tmp = int(args.new_width)
except ValueError:
    print("Width value is not an integer.")
    sys.exit(1)

try:
    tmp = int(args.new_height)
except ValueError:
    print("Height value is not an integer.")
    sys.exit(1)

if args.extension not in ["jpg", "jpeg", "png", "webp"]:
    print("Extension is not valid.")
    sys.exit(1)

try:
    if args.index != None:
        index = int(args.index)
    else:
        index = 0
except ValueError:
    print("Index must be integer.")
    sys.exit(1)

        
# for img in os.listdir(args.source):
#     if os.path.isfile(os.path.join(args.source, img)) and ".xml" not in img:
#         img_name = img[:-4]

#         tree = ET.parse(os.path.join(args.source, img_name + ".xml"))
#         root = tree.getroot()

#         img = Image.open(os.path.join(args.source, img))
#         img = img.convert('RGB')
#         img = img.resize((int(args.width), int(args.height)))
        
#         new_file_name = args.name + str(index)
        
#         for obj in root.findall('filename'):
#             obj.text = new_file_name + '.jpg'

#         for obj in root.findall('path'):
#             obj.text = new_file_name + '.jpg'

#         img.save(os.path.join(args.destination, new_file_name + '.jpg'))
#         tree.write(os.path.join(args.destination, new_file_name + ".xml"))

#         index += 1

print(args.xml)
print("Succesfully preprocessed images.")
