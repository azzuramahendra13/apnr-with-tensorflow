import xml.etree.ElementTree as ET
import os

FOLDER_PATH = os.path.abspath('workspace/training_demo/images/train')

for file in os.listdir(FOLDER_PATH):
    if '.xml' in file:
        tree = ET.parse(os.path.join(FOLDER_PATH, file))
        root = tree.getroot()

        for obj in root.findall('object'):
            if obj.find('name').text in ['Plat Kendaraan', 'plat', 'plat-nomor', 'platNomor', 'plate']:
                obj.find('name').text = 'plat-nomor'
            else:
                root.remove(obj)

        tree.write(os.path.join(FOLDER_PATH, file))
