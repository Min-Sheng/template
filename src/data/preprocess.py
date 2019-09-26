import os
import random
import glob
import imageio
import json
import logging
import argparse
import numpy as np
import cv2
import scipy.ndimage.morphology as ndi_morph
import scipy.ndimage.measurements as measurements
import math

def create_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)

def split_subdataset(subdataset_dir, subdataset, seed = 448):
    img_name_list = os.listdir(subdataset_dir)
    random.seed(seed)
    random.shuffle(img_name_list)
    train = img_name_list[:int(len(img_name_list)*0.8)]
    valid = img_name_list[int(len(img_name_list)*0.8):int(len(img_name_list)*0.9)]
    test = img_name_list[int(len(img_name_list)*0.9):]
    data_list = {'train': train, 'train_full': train[:int(len(train)*0.5)], 'train_weak': train[int(len(train)*0.5):], 'val': valid, 'test': test}
    with open('../data/{:s}/train_val_test.json'.format(subdataset), 'w') as file:
        json.dump(data_list, file)
        
def split_dataset(dataset_dir, dataset, seed = 448):
    subdataset_list = glob.glob(dataset_dir+'*')
    img_name_list=[]
    for subdataset in subdataset_list:
        if subdataset.split('/')[-1] in ['stage_1_test', 'nuclei_partial_annotations']:
            continue
        temp = glob.glob(subdataset+'/*')
        img_name_list.extend(temp)

    img_name_list = [item.split('/')[-2]+'/'+ item.split('/')[-1] for item in img_name_list]
    random.seed(seed)
    random.shuffle(img_name_list)
    train = img_name_list[:int(len(img_name_list)*0.8)]
    valid = img_name_list[int(len(img_name_list)*0.8):int(len(img_name_list)*0.9)]
    test = img_name_list[int(len(img_name_list)*0.9):]
    data_list = {'train': train, 'train_full': train[:int(len(train)*0.5)], 'train_weak': train[int(len(train)*0.5):], 'val': valid, 'test': test}
    with open('../data/{:s}/All/train_val_test.json'.format(dataset), 'w') as file:
        json.dump(data_list, file)
        
def create_labels_instance_for_subdataset(subdataset_dir):
    img_name_list = os.listdir(subdataset_dir)
    for img_name in img_name_list:
        mask_list = glob.glob(os.path.join(subdataset_dir, img_name, 'masks','*.png'))
        masks = []
        for i , mask_file in enumerate(mask_list):
            mask = imageio.imread(mask_file)
            masks.append(mask*(i+1))
        masks = np.stack(masks, axis = -1).astype(np.int32)
        masks = np.sum(masks, axis = -1)
        create_folder(os.path.join(subdataset_dir, img_name, 'label'))
        np.save('{:s}/{:s}_label.npy'.format(os.path.join(subdataset_dir, img_name, 'label'), img_name), masks.astype(np.int32))
            
def create_labels_instance_for_dataset(dataset_dir):
    subdataset_dir_list = glob.glob(dataset_dir+'*')
    for subdataset_dir in subdataset_dir_list:
        create_labels_instance_for_subdataset(subdataset_dir)

def create_three_cls_label_for_subdataset(subdataset_dir):
    img_name_list = os.listdir(subdataset_dir)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    for img_name in img_name_list:
        mask_list = sorted(glob.glob(os.path.join(subdataset_dir, img_name, 'masks','*.png')))
        print(img_name)
        print(len(mask_list))
        mask = imageio.imread(mask_list[0])
        contour = cv2.dilate(mask, kernel, iterations=1) - mask
        masks_w_e = mask.copy()
        instance_mask = mask.copy()
        masks_w_e[np.where(mask)] = 2
        masks_w_e[np.where(contour)] = 1
        instance_mask[np.where(mask)] = 1
        for i , mask_file in enumerate(mask_list[1:]):
            mask = imageio.imread(mask_file)
            contour = cv2.dilate(mask, kernel, iterations=1) - mask
            masks_w_e[np.where(mask)]=2
            masks_w_e[np.where(contour)]=1
            instance_mask[np.where(mask)] = (i+2)
        create_folder(os.path.join(subdataset_dir, img_name, '3cls_label'))
        create_folder(os.path.join(subdataset_dir, img_name, 'no_overlap_label'))
        np.save('{:s}/{:s}_3cls_label.npy'.format(os.path.join(subdataset_dir, img_name, '3cls_label'), img_name), masks_w_e.astype(np.uint8))
        np.save('{:s}/{:s}_no_overlap_label.npy'.format(os.path.join(subdataset_dir, img_name, 'no_overlap_label'), img_name), instance_mask.astype(np.uint8))

def vec_to_angle(vector):
    a = (np.arctan2(vector[..., 0], vector[..., 1]) / math.pi + 1) / 2
    a = a /np.max(a)
    r = np.sqrt(vector[..., 0] ** 2 + vector[..., 1] ** 2)
    r = -r + np.max(r)
    r = r / np.max(r)
    return np.stack((a,r), axis=-1)

def create_watershed_direction_and_energy_for_subdataset(subdataset_dir):
    img_name_list = os.listdir(subdataset_dir)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    res_channels = 3
    for img_name in img_name_list:
        mask_list = sorted(glob.glob(os.path.join(subdataset_dir, img_name, 'masks','*.png')))
        print(img_name)
        print(len(mask_list))
        mask = imageio.imread(mask_list[0])
        rows, cols = mask.shape
        disp_field = np.zeros((rows, cols, res_channels))
        center_of_mass = measurements.center_of_mass(mask)
        current_offset_field = np.zeros((rows, cols, 2))
        current_offset_field[:, :, 0] = np.expand_dims(center_of_mass[0] - np.arange(0, rows), axis=1)
        current_offset_field[:, :, 1] = np.expand_dims(center_of_mass[1] - np.arange(0, cols), axis=0)
        strength = (np.sqrt(current_offset_field[:, :, 0]**2 + current_offset_field[:, :, 1]**2)) + 1e-10
        disp_field[:, :, 0:2][mask > 0] = (current_offset_field[:, :, 0:2]/strength[:,:,None])[mask > 0]
        strength = -strength + np.max(strength[mask>0])
        strength = strength / (np.max(strength) + 1e-10)
        #disp_field[:, :, 2][mask>0] = strength[mask>0]
        #disp_field[int(round(center_of_mass[0])),int(round(center_of_mass[1])),2] = 1
        disp_field[:, :, 2][mask>0] = 1
        #center_point = cv2.circle(disp_field[:, :, 2].astype(np.uint8),
        #                          (int(round(center_of_mass[1])),int(round(center_of_mass[0]))), 2, 2, -1)
        #disp_field[:, :, 2] = center_point.astype(np.float32)
        disp_field[:, :, 2][np.where(strength>0.85)] = 2
        if np.isnan(strength).any():
            raise Exception("NaN!")
        for i , mask_file in enumerate(mask_list[1:]):
            mask = imageio.imread(mask_file)
            rows, cols = mask.shape
            center_of_mass = measurements.center_of_mass(mask)
            current_offset_field = np.zeros((rows, cols, 2))
            current_offset_field[:, :, 0] = np.expand_dims(center_of_mass[0] - np.arange(0, rows), axis=1)
            current_offset_field[:, :, 1] = np.expand_dims(center_of_mass[1] - np.arange(0, cols), axis=0)
            strength = (np.sqrt(current_offset_field[:, :, 0]**2 + current_offset_field[:, :, 1]**2)) + 1e-10
            disp_field[:, :, 0:2][mask > 0] = (current_offset_field[:, :, 0:2]/strength[:,:,None])[mask > 0]
            strength = -strength + np.max(strength[mask>0])
            strength = strength / (np.max(strength) + 1e-10)
            #disp_field[:, :, 2][mask>0] = strength[mask>0]
            #disp_field[int(round(center_of_mass[0])),int(round(center_of_mass[1])),2] = 1
            disp_field[:, :, 2][mask>0] = 1
            #center_point = cv2.circle(disp_field[:, :, 2].astype(np.uint8),
            #              (int(round(center_of_mass[1])),int(round(center_of_mass[0]))), 2, 2, -1)
            #disp_field[:, :, 2] = center_point.astype(np.float32)
            disp_field[:, :, 2][np.where(strength>0.85)] = 2
            if np.isnan(strength).any():
                raise Exception("NaN!")
        #disp_field[:,:,2] = cv2.dilate(disp_field[:,:,2], kernel, iterations=2) 
        create_folder(os.path.join(subdataset_dir, img_name, 'watershed_label'))
        np.save('{:s}/{:s}_watershed_label.npy'.format(os.path.join(subdataset_dir, img_name, 'watershed_label'), img_name), disp_field.astype(np.float32))

def create_three_cls_label_for_dataset(dataset_dir):
    print("Create three cls label")
    subdataset_dir_list = glob.glob(dataset_dir+'*')
    for subdataset_dir in subdataset_dir_list:
        create_three_cls_label_for_subdataset(subdataset_dir)

def create_watershed_direction_and_energy_for_dataset(dataset_dir):
    print("Create watershed direction and energy")
    subdataset_dir_list = glob.glob(dataset_dir+'*')
    for subdataset_dir in subdataset_dir_list:
        print(subdataset_dir)
        create_watershed_direction_and_energy_for_subdataset(subdataset_dir)

def main(args):
    data_dir = args.data_dir
    data_split_dir = args.data_split_dir
    print(data_dir)
    print(data_split_dir)
    create_folder(data_split_dir)
    if not args.merge_all:
        subdataset_dir = data_dir
        dataset = subdataset_dir.split('/')[-2]
        subdataset = subdataset_dir.split('/')[-1]
        create_folder(data_split_dir + '/' + dataset)
        create_folder(data_split_dir + '/' + dataset+'/' + subdataset)
        #split_subdataset(subdataset_dir, dataset + '/' + subdataset)
        #create_labels_instance_for_subdataset(subdataset_dir)
        create_three_cls_label_for_subdataset(subdataset_dir)
        create_watershed_direction_and_energy_for_subdataset(subdataset_dir)
        
    else:
        dataset_dir = data_dir
        dataset = dataset_dir.split('/')[-1]
        create_folder(data_split_dir + '/' + dataset)
        create_folder(data_split_dir + '/' + dataset + '/All')
        #split_dataset(dataset_dir, dataset)
        #create_labels_instance_for_dataset(dataset_dir)
        #create_three_cls_label_for_dataset(dataset_dir)
        create_watershed_direction_and_energy_for_dataset(dataset_dir)

def _parse_args():
    parser = argparse.ArgumentParser(description="The data preprocessing.")
    parser.add_argument('data_dir', type=str, help='The directory of the dataset.')
    parser.add_argument('--data_split_dir', type=str, default='/home/vincentwu-cmlab/template/data_split/', help='The directory of the data split file.')
    parser.add_argument('--merge_all', action='store_true', help='Merge all data from each subdataset into one data split file.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
