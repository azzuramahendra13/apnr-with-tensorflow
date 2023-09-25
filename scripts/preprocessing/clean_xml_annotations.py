import xml.etree.ElementTree as ET
import os

FOLDER_PATH = '../../workspace/training_dataset_2/images/dataset_3/test'

for file in os.listdir(FOLDER_PATH):
    if '.xml' in file:
        tree = ET.parse(os.path.join(FOLDER_PATH, file))
        root = tree.getroot()

        for obj in root.findall('filename'):
            obj.text = file[:-4] + '.jpg'

        for obj in root.findall('path'):
            obj.text = file[:-4] + '.jpg'

        tree.write(os.path.join(FOLDER_PATH, file))
