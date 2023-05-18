from PIL import Image
from PIL import ImageDraw
from bs4 import BeautifulSoup


def render_one(img):
    img = Image.open(r'data\PascalVOC\TrainVal\VOCdevkit\VOC2011\JPEGImages\2008_000054.jpg')
    with open(r'data\PascalVOC\annotations\bird\2008_000054_2.xml') as f:
        data = f.read()

    soup_data = BeautifulSoup(data, "xml")
    keypoints = soup_data.find_all('keypoint')

    print(keypoints)

    labels = []
    for keypoint in keypoints:
        labels.append((keypoint['name'], int(float(keypoint['x'])), int(float(keypoint['y'])), int(float(keypoint['visible']))))

    draw = ImageDraw.Draw(img)

    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown', 'grey', 'black', 'white']
    circle_size = 3
    for label, color in zip(labels, colors):
        draw.ellipse((label[1], label[2], label[1]+circle_size, label[2]+circle_size), width=5, fill=color)
        draw.text((label[1]+40, label[2]), label[0], fill=color)

    img.show()

