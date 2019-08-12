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
        mask_list = glob.glob(os.path.join(subdataset_dir, img_name, 'masks','*.png'))
        masks = []
        contours = []
        for i , mask_file in enumerate(mask_list):
            mask = imageio.imread(mask_file)
            #cnts,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            #contour = np.zeros(mask.shape)
            #cv2.drawContours(contour, cnts, 0, 1, 1)
            #contour = cv2.morphologyEx(contour, cv2.MORPH_CLOSE, kernel)
            contour = cv2.dilate(mask, kernel, iterations=1) - mask
            masks.append(mask)
            contours.append(contour)
        masks = np.stack(masks, axis = -1).astype(np.uint8)
        contours = np.stack(contours, axis = -1).astype(np.uint8)
        masks = (np.sum(masks, axis = -1)>0).astype(np.uint8)
        contours = (np.sum(contours, axis = -1)>0).astype(np.uint8)
        masks_w_e = masks + contours*2
        masks_w_e[np.where(masks_w_e>1)]=3
        masks_w_e[np.where(masks_w_e==1)]=2
        masks_w_e[np.where(masks_w_e==3)]=1
        create_folder(os.path.join(subdataset_dir, img_name, '3cls_label'))
        np.save('{:s}/{:s}_3cls_label.npy'.format(os.path.join(subdataset_dir, img_name, '3cls_label'), img_name), masks_w_e.astype(np.uint8))
        
def create_watershed_direction_and_energy_for_subdataset(subdataset_dir):
    
    img_name_list = os.listdir(subdataset_dir)
    res_channels = 3
    
    for img_name in img_name_list:
        mask_list = glob.glob(os.path.join(subdataset_dir, img_name, 'masks','*.png'))
        masks = []
        #watershed = []
        disp = []
        
        for i , mask_file in enumerate(mask_list):
            mask = imageio.imread(mask_file)
            rows, cols = mask.shape
            res = np.zeros((rows, cols, res_channels))
            
            #edt, inds = distance_transform_edt(mask, return_distances=True, return_indices=True)
            #border_vector = np.array([
            #np.expand_dims(np.arange(0, rows), axis=1) - inds[0],
            #np.expand_dims(np.arange(0, cols), axis=0) - inds[1]])
            #border_vector_norm = border_vector / (np.linalg.norm(border_vector, axis=0, keepdims=True) + 1e-5)
            #edt_norm = (edt - edt.min()) / (edt.max() - edt.min())

            #res[:, :, 0] = border_vector_norm[0]
            #res[:, :, 1] = border_vector_norm[1]
            #res[:, :, 2] = edt
            
            center_of_mass = measurements.center_of_mass(mask)

            current_offset_field = np.zeros((rows, cols, 2))
            current_offset_field[:, :, 0] = np.expand_dims(center_of_mass[0] - np.arange(0, rows), axis=1)
            current_offset_field[:, :, 1] = np.expand_dims(center_of_mass[1] - np.arange(0, cols), axis=0)
           
            #res[:, :, 0][mask > 0] = 2*(current_offset_field[:, :, 0][mask > 0]  - current_offset_field[:, :, 0][mask > 0].min())/\
            #                        (current_offset_field[:, :, 0][mask > 0].max() - current_offset_field[:, :, 0][mask > 0].min()+ 1e-5)-1 + 1e-5
            #res[:, :, 1][mask > 0] = 2*(current_offset_field[:, :, 1][mask > 0]  - current_offset_field[:, :, 1][mask > 0].min())/\
            #                        (current_offset_field[:, :, 1][mask > 0].max() - current_offset_field[:, :, 1][mask > 0].min()+ 1e-5)-1 + 1e-5
            res[:, :, 0:2][mask > 0] = current_offset_field[:, :, 0:2][mask > 0]
            res[:, :, 2] = mask
            masks.append(mask)
            #watershed.append(res)
            disp.append(res)
        
        masks = np.stack(masks, axis = -1).astype(np.uint8)
        masks = (np.sum(masks, axis = -1)>0).astype(np.uint8)
        #watershed = np.stack(watershed, axis = -1).astype(np.float32)
        #watershed = np.sum(watershed, axis = -1).astype(np.float32)
        disp_field = np.stack(disp, axis = -1).astype(np.float32)
        disp_field = np.sum(disp_field, axis = -1).astype(np.float32)
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
        create_three_cls_label_for_dataset(dataset_dir)
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