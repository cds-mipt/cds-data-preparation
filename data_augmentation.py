import imgaug as ia
import argparse
import imgaug.augmenters as iaa
from shapely.geometry import Polygon
from cvat import CvatDataset
import shapely
import numpy as np
from urllib.request import urlopen
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw
import random
import garbage as g
from tqdm import tqdm

import albumentations as albu
import numpy as np
import cv2
import png
import os

main_folder = "/media/andreizoltan/DATA/Ice_Vision/dataset/background/only1and5/"
clear_folder = "/media/andreizoltan/DATA/Ice_Vision/dataset/background/clear-1-5/"

num = 0
def save_image(data, folder, name, image_name) :
    image_name = image_name.split('.')[0]
    im = Image.fromarray(data)
    global num
    folder_a = args.output_folder
    print(folder_a+folder+"/"+image_name + "_" + str(num).zfill(4)+name)
    im.save(folder_a+folder+"/"+image_name + "_" + str(num).zfill(4)+name)
    if folder == "clear":
        num+= 1
############################################################################################################

def label_image(label):
    return "car"

def points_please(input_file):
    ds = CvatDataset()
    ds.load(input_file)
    polygons = list()
    all_polygons = list()
    for image_id in ds.get_image_ids():
        for polygon in ds.get_polygons(image_id):
            label = label_image(polygon["label"].replace("_", "."))
            polygons += [polygon["points"]]
        for i in range(len(polygons)):
            for j in range(len(polygons[i])):
                polygons[i][j] = tuple(polygons[i][j])
        all_polygons.append(polygons)
        polygons = list()
    return all_polygons

ww = 1200


def masks_please(points, width, height):
    img = Image.new('L', (width, height), 0)

    for i in range(len(points)):
        f = list()
        for j in range(len(points[i])):
            f.append(tuple(points[i][j]))
        ImageDraw.Draw(img).polygon(f, outline=255, fill=255)
    mask = np.array(img)
    return mask
#=======================================================================================

def pol_to_dots(poly):
    l = list()
    x, y = poly.exterior.coords.xy
    coord = [x, y]
    for i in range(len(coord[0])):
        xy = [coord[0][i], coord[1][i]]
        dots = tuple(xy)
        l.append(dots)
    return l

def checked(pol):
    global ww
    pol = Polygon(pol)
    p1 = Polygon([(0, 0), (0, 1200), (1200, 1200), (1200, 0)])
    p1 = p1.intersection(pol)
    if type(p1) == shapely.geometry.multipolygon.MultiPolygon:
        return 0
    if p1.is_empty == True:
        return 0
    else: return pol_to_dots(p1)


def augment_please(s_images, s_images_l, masks, points, image_name):
    batches = 40
    global ww
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    full_list = list()
    for i in range(batches):
        aug = iaa.Sequential([
            iaa.Affine(rotate=(-10, 10)),
            iaa.CropToFixedSize(width=ww, height=ww),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.10)))
        ])
        aug = aug.to_deterministic()
        batch_aug = aug.augment(
            images=s_images, polygons=points,
            return_batch=True)
        batch_aug_l = aug.augment(
            images=s_images_l, polygons=points,
            return_batch=True)
        batch_aug_masks = aug.augment(
            images=masks, polygons=points,
            return_batch=True)


        new_images = batch_aug.images_aug
        new_images_l = batch_aug_l.images_aug
        new_masks = batch_aug_masks.images_aug

        new_images_l = np.asarray(new_images_l)
        new_images = np.asarray(new_images)

        for j in range(len(new_images)):
            save_image(new_images[j], "cars", ".jpg", image_name)
            save_image(new_masks[j], "masks", ".bmp", image_name)
            save_image(new_images_l[j], "clear", ".jpg", image_name)

        new_list = list()

        new_polygo = batch_aug.polygons_aug
        new_polygo = np.asarray(new_polygo)


        for p in new_polygo:
            for polygon in p:
                if checked(polygon) == 0:
                    continue
                else:
                    new_list.append(checked(polygon))
            full_list.append(new_list)
            new_list = list()

        ww = 1200

    return full_list



def get_polygons(points):
    full_list = list()
    batch = 5
    for i, image_name in enumerate(os.listdir(args.initial_folder)):
        image        = mpimg.imread(args.initial_folder+image_name)
        image_double = mpimg.imread(args.clear_folder+image_name)
        width = len(image[0])
        height = len(image)
        print(i, image_name)
        mask = masks_please(points[i], width, height)
        s_images = [image] * batch
        s_images_l = [image_double] * batch
        masks = [mask] * batch
        points_k = [points[i]] * batch

        full = augment_please(s_images, s_images_l, masks, points_k, image_name)
        full_list.extend(full)
    full_list = np.asarray(full_list)
    POINTSS = full_list
    return POINTSS


def build_parser():
    parser = argparse.ArgumentParser("Add polygons according to sign class")
    parser.add_argument(
        "--input-file",
        type=str
    )
    parser.add_argument(
        "--output-file",
        type=str
    )
    parser.add_argument(
        "--initial-folder",
        type=str
    )
    parser.add_argument(
        "--clear-folder",
        type=str
    )
    parser.add_argument(
        "--output-folder",
        type=str
    )
    return parser

def dump(polygons, output_file):
    ds = CvatDataset()
    image_id = 0
    for POINTS in polygons:
        ds.add_image(image_id)
        for points in POINTS:
            ds.add_polygon(
                image_id=image_id,
                points=points,
                label="car",
                occluded=0)
        image_id += 1
        print(image_id)
    ds.dump(output_file)


def main(args):
    points = points_please(args.input_file)
    polygons = get_polygons(points)
    dump(polygons, args.output_file)

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)