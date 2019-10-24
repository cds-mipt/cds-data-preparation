import os
import pandas as pd
import argparse
import json


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type=str,
        help="input file in json format"
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default='',
        help="folder for output files"
    )
    parser.add_argument(
        "--ann-name",
        type=str,
        default='',
        help="name of city"
    )
    parser.add_argument(
        "--attitude",
        type=str,
        default='0.8',
        help="attitude of train and test"
    )
    return parser


def get_filename(path):
    #filename = args.dataset_folder + '/' + path.split('/')[-1]
    filename = path.replace(r'/scratch/fs2/DTLD_final/', '').replace(r'tiff', 'png')
    return filename


def first_digit(category):
    category = str(category)
    #return "traffic_light"
    return category[4]


def get_classes(df):
    classes = set()
    for index, row in df.iterrows():
        for i in range(len(row['objects'])):
            object = row['objects'][i]
            classes.add(first_digit(object['class_id']))
    return classes


def get_images(df):
    data['images'] = list()
    for index, row in df.iterrows():
        frame = dict()
        frame['file_name'] = get_filename(row['path'])
        frame['height'] = 1024
        frame['width'] =  2048
        frame['id'] = index
        data['images'].append(frame)


def get_annotations(df):
    data['annotations'] = list()
    for index, row in df.iterrows():
        for i in range(len(row['objects'])):
            object = row['objects'][i]
            frame = dict()
            frame['area'] = int( object['width'] * object['height'])
            frame['iscrowd'] = 0
            frame['image_id'] = index
            frame['bbox'] = [int(object['x']), \
                             int(object['y']), \
                             int(object['width']), \
                             int(object['height'])]
            frame['category_id'] = first_digit(object['class_id'])
            frame['id'] = object['unique_id']
            frame['ignore'] = 0
            frame['segmentation'] = []
            data['annotations'].append(frame)


def get_categories(classes):
    data['categories'] = list()
    #classes = get_classes(df)
    for each_class in classes:
        category = dict()
        category['supercategory'] = "none"
        category['id'] = each_class
        category['name'] = each_class
        data['categories'].append(category)

data = {}
def main(args):
    df = pd.read_json(args.input_file)
    del df['disp_path']
    classes = get_classes(df)
    print(df.shape)
    train_count = int(df.shape[0] * float(args.attitude))
    train = df[:train_count] # set the number of train images
    test = df[train_count:] #  set the number of test images



    get_images(train)
    data['type'] = "instances"
    get_annotations(train)
    get_categories(classes)
    with open(args.output_folder + args.ann_name + '_ann_train.json', 'w') as outfile:
        json.dump(data, outfile)

    data.clear()
    get_images(test)
    data['type'] = "instances"
    get_annotations(test)
    get_categories(classes)
    with open(args.output_folder + args.ann_name + '_ann_test.json', 'w') as outfile:
        json.dump(data, outfile)

    print(args.output_folder + args.ann_name + '_ann_train.json')
    print(args.output_folder + args.ann_name + '_ann_test.json')
if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
