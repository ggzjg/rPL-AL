import skimage
import matplotlib.pyplot as plt
import createData
import net
import torch
from torchvision import transforms
import copy
import numpy as np
import argparse
import scipy.io as sio
from createData import applyPCA, random_index, feature_normalize, indexToCube_label
from skimage import measure


def neighbor_score(center_index, pre_score, neig_stride):
    score_sum = 0
    for i in range(-neig_stride, neig_stride + 1):
        for j in range(-neig_stride, neig_stride + 1):
                if i != 0 or j != 0:
                    #score = pre_score[center_index[0] + i, center_index[1] + j]
                    score_sum += pre_score[center_index[0] + i, center_index[1] + j]

    return score_sum


def pixel_fine_class(area_train_index, area_train_label, pseudo_ind2score, connect_area_index, selected_connect_map):
    for i in range(len(area_train_index)):
        tmp = np.zeros_like(pseudo_ind2score[list(pseudo_ind2score.keys())[0]])
        tmp[area_train_label[i]-1] = 1
        pseudo_ind2score.update({area_train_index[i]: tmp})
    unique_label = np.unique(area_train_label)
    for i in range(len(connect_area_index)):
        new_label = unique_label[np.argmax(pseudo_ind2score[connect_area_index[i]][unique_label-1])]
        selected_connect_map[connect_area_index[i]] = new_label
    return selected_connect_map


if __name__ == '__main__':
    dataset_names = ['IP', 'SA', 'PU']
    parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                                 " various hyperspectral datasets")
    parser.add_argument('--dataset', type=str, default='PU', choices=dataset_names,
                        help="Dataset to use.")
    parser.add_argument('--perclass', type=int, default=5)  # few-shot number
    parser.add_argument('--device', type=str, default="cuda:0", choices=("cuda:0", "cuda:1"))
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--windowSize', type=int, default=13)
    parser.add_argument('--numComponents', type=int, default=30)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--classi_lr', type=float, default=1e-3)
    parser.add_argument('--train', type=bool, default=1)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--iter_idx', type=str, default='2')

    args = parser.parse_args()
    epoch, dataset, windowSize, numComponents, perclass = args.epoch, args.dataset, \
                                                          args.windowSize, args.numComponents, args.perclass
    output_units = 9 if (dataset == 'PU' or dataset == 'PC') else 16  # 类别数目
    stratify = np.arange(0, output_units, 1)
    hsi_data, gt_map = createData.loadData(dataset)  # 145*145*200, 145*145
    X = createData.applyPCA(hsi_data, numComponents, windowSize)
    X = createData.feature_normalize(X)

    train_index = np.load('./Split_Data/{}_{}_shot_train_index.npy'.format(args.dataset, args.perclass),allow_pickle=True).tolist()
    for i in range(len(train_index)): train_index[i] = tuple(train_index[i])  # to tuple
    conf_index = np.load('./Split_Data/pseudo_thres_index_fs'+str(args.perclass)+'_iter'+args.iter_idx+'.npy',allow_pickle=True)
    conf_index2label = np.load('./Split_Data/pseudo_ind2label_fs'+str(args.perclass)+'_iter'+args.iter_idx+'.npy',allow_pickle=True).tolist()
    pseudo_ind2score = np.load('./Split_Data/pseudo_ind2score_fs'+str(args.perclass)+'_iter'+args.iter_idx+'.npy',allow_pickle=True).tolist()
    conf_img_scores = [0.0] * np.zeros_like(gt_map)
    conf_img_label = np.zeros_like(gt_map)
    select_con_map = np.zeros_like(gt_map)

    # confidence score map
    for index, value in pseudo_ind2score.items():
        score = max(value)
        conf_img_scores[index[0], index[1]] = score

    # confidence label map
    for index, label in conf_index2label.items():
        conf_img_label[index[0], index[1]] = label + 1
    for i in range(len(train_index)):
        conf_img_label[train_index[i][0], train_index[i][1]] = gt_map[train_index[i][0], train_index[i][1]]


    connect_map, connect_area_num = skimage.measure.label(np.int32(conf_img_label>0), connectivity=2, background=0, return_num=True)
    selected_connect_map = np.zeros_like(connect_map)

    for i in range(connect_area_num):
        # get ith connect area pixel index
        connect_area_index = np.where(connect_map == i + 1)  # connect_map background = 0
        connect_area_index = list(zip(connect_area_index[0], connect_area_index[1]))
        # get common index
        com_index = list(set(train_index) & set(connect_area_index))
        area_train_label = []
        sum_score = []
        if com_index != []:
            # get common index label
            for j in range(len(com_index)):
                area_train_label.append(gt_map[com_index[j][0], com_index[j][1]])
            # common index have different label
            if len(list(set(area_train_label))) > 1:
                selected_connect_map = pixel_fine_class(com_index, area_train_label, pseudo_ind2score,
                                                        connect_area_index, selected_connect_map)
                # per index neighbor sum score
                # for ind in range(len(com_index)):
                #     neig_sum_score = neighbor_score(com_index[ind], conf_img_scores, neig_stride=1)
                #     sum_score.append(neig_sum_score)
                # get max score index
                # max_score_index = sum_score.index(max(sum_score))
                # IndexError: list index out of range
                # select_label = area_train_label[max_score_index]
                # selected_connect_map[connect_map == i + 1] = select_label
            else:
                selected_connect_map[connect_map == i + 1] = area_train_label[0]
        else:
            # delete connect area
            connect_map[connect_map == i + 1] = 0

    # get final index
    final_connect_index = list(zip(np.nonzero(selected_connect_map)[0], np.nonzero(selected_connect_map)[1]))
    np.save('./Split_Data/final_connect_pseudolabel_fs'+str(args.perclass)+'_iter'+args.iter_idx+'.npy', selected_connect_map)
    np.save('./Split_Data/final_connect_index_fs'+str(args.perclass)+'_iter'+args.iter_idx+'.npy', final_connect_index)

    sio.savemat('./Split_Data/final_connect_pseudolabel_fs'+str(args.perclass)+
                '_iter'+args.iter_idx+'.mat', {'connect_map':selected_connect_map})

    sio.savemat('./Split_Data/final_connect_index_fs' + str(args.perclass) +
                '_iter' + args.iter_idx + '.mat', {'final_connect_index': final_connect_index})

    sio.savemat('./Split_Data/conf_img_label_fs' + str(args.perclass) +
                '_iter' + args.iter_idx + '.mat', {'conf_img_label': conf_img_label})

    print('final_connect_index finished')
