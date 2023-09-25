import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("source", help="Image source folder")

args = parser.parse_args()

for file in os.listdir(args.source):
     if ".xml" in file and not os.path.exists(os.path.join(args.source, file[:-4] + ".jpg")):
          os.remove(os.path.join(args.source, file))