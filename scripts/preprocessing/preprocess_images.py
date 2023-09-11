import argparse
from PIL import Image 
import os
import sys

parser = argparse.ArgumentParser()

parser.add_argument("source", help="Image source folder")
parser.add_argument("destination", help="Image destination folder")
parser.add_argument("width", help="Image width")
parser.add_argument("height", help="Image height")
parser.add_argument("extension", help="Image extension")
parser.add_argument("name", help="Image name")
parser.add_argument("-i", "--index", help="Starting index for naming")

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
    tmp = int(args.width)
except ValueError:
    print("Width value is not an integer.")
    sys.exit(1)

try:
    tmp = int(args.height)
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

        
for img in os.listdir(args.source):
    if os.path.isfile(os.path.join(args.source, img)):
        img = Image.open(os.path.join(args.source, img))
        img = img.convert('RGB')
        img = img.resize((int(args.width), int(args.height)))
        
        new_file_name = args.name + str(index) + '.' + args.extension
        
        img.save(os.path.join(args.destination, new_file_name))

        index += 1

print("Succesfully preprocessed images.")
