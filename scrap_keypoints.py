import json
import os
import csv


def extract_keypoints():
    data_id_dir = os.path.join('data', 'PascalVOC', 'data-[256, 256]-pca.json')

    with open(data_id_dir) as f:
        data_ids = json.load(f)

    classes = []
    output = {}

    for entity in data_ids:
        cls = data_ids[entity]['cls']

        if cls not in classes:
            print('added class')
            classes.append(cls)
            output[cls] = []

        kpts = data_ids[entity]['kpts']

        for kpt in kpts:
            label = kpt['labels']
            if label not in output[cls]:
                output[cls].append(label)

    max_kpt_examples = {}
    for entity in data_ids:
        cls = data_ids[entity]['cls']
        kpts = data_ids[entity]['kpts']

        if cls not in max_kpt_examples:
            max_kpt_examples[cls] = []
        if len(kpts) == len(output[cls]):
            max_kpt_examples[cls].append(entity)

    print(max_kpt_examples)

    values = list(max_kpt_examples.values())

    # Determine the maximum length of the value lists
    max_length = max(len(value) for value in values)

    # Fill the value lists with empty strings if necessary
    filled_values = [value + [''] * (max_length - len(value)) for value in values]

    # Transpose the values to align them with the keys as columns
    transposed_values = list(zip(*filled_values))

    filename = 'max_keypoints.csv'
    # Write the data to the CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row with the keys
        writer.writerow(classes)

        # Write the rows with the transposed values
        writer.writerows(transposed_values)




def find_id():  #with max keypoints
    data_id_dir = os.path.join('data', 'PascalVOC', 'data-[256, 256]-pca.json')

    with open(data_id_dir) as f:
        data_ids = json.load(f)

    for entity in data_ids:
        if data_ids[entity]['cls'] == 'aeroplane' and len(data_ids[entity]['kpts']) == 16:
            print(entity)


extract_keypoints()