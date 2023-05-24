from PIL import Image
from PIL import ImageDraw
from bs4 import BeautifulSoup
import os


def add_labels(img_src, img_data):
    soup_data = BeautifulSoup(img_data, "xml")
    keypoints = soup_data.find_all('keypoint')

    # print(keypoints)

    labels = []
    for keypoint in keypoints:
        labels.append(
            (keypoint['name'], int(float(keypoint['x'])), int(float(keypoint['y'])), int(float(keypoint['visible']))))

    draw = ImageDraw.Draw(img_src)

    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown', 'grey', 'black', 'white']
    circle_size = 3
    padding = 0
    for label, color in zip(labels, colors):
        draw.ellipse((label[1], label[2], label[1] + circle_size, label[2] + circle_size), width=5, fill=color)
        draw.text((0, 0+padding), label[0], fill=color) # , stroke_width=2, stroke_fill='green'
        padding += 20
    return img_src

def render_one(img):
    img_path = os.path.join('data', 'PascalVOC', 'TrainVal', 'VOCdevkit', 'VOC2011', 'JPEGImages', '2008_000043.jpg')
    label_path = os.path.join('data', 'PascalVOC', 'annotations', 'chair', '2008_000043_2.xml')
    img = Image.open(img_path)
    with open(label_path) as f:
        data = f.read()

    # pip install lxml
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

    # apt-get update
    # apt-get install imagemagick
    img.show()


def render_pair(img1, img2):
    img1_class = img1.split('_')[-1]
    img2_class = img2.split('_')[-1]

    img1_labels = img1[0:img1.rfind('_')]
    img2_labels = img2[0:img2.rfind('_')]

    if img1_labels.count('_') > 1:
        img1_id = img1[0:img1_labels.rfind('_')]
    else:
        img1_id = img1_labels

    # print(f'img1_id: {img1_id}\nimg1_src: {img1_labels}')
    if img2_labels.count('_') > 1:
        img2_id = img2[0:img2_labels.rfind('_')]
    else:
        img2_id = img2_labels

    img1_src = Image.open(os.path.join('data', 'PascalVOC', 'TrainVal', 'VOCdevkit', 'VOC2011', 'JPEGImages',
                                       f'{img1_id}.jpg'))
    with open(os.path.join('data', 'PascalVOC', 'annotations', img1_class, f'{img1_labels}.xml')) as f:
        data1 = f.read()

    img2_src = Image.open(os.path.join('data', 'PascalVOC', 'TrainVal', 'VOCdevkit', 'VOC2011', 'JPEGImages',
                                       f'{img2_id}.jpg'))
    with open(os.path.join('data', 'PascalVOC', 'annotations', img2_class, f'{img2_labels}.xml')) as f:
        data2 = f.read()
    add_labels(img1_src, data1).save('img1.jpg')
    add_labels(img2_src, data2).save('img2.jpg')
