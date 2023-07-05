import os
import re
pattern = r'\d+(.*)\.png'
IMAGE_DIR = os.path.dirname(__file__) + '/img/'
class_names = ['left', 'right', 'mid']
count = [0,0,0]
for image_name in os.listdir(IMAGE_DIR):
    label = re.findall(pattern, image_name)[0]
    count[class_names.index(label)] = count[class_names.index(label)] + 1

print(f'''
      LEFT = {count[0]}
      RIGHT = {count[1]}
      MID = {count[2]}
      ''')

