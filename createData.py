''' create few shot loader '''
import random
import numpy as np
import scipy.io as sio
import torch
from sklearn.decomposition import PCA
import os
from operator import truediv

def loadData(name):
    data_path = os.path.join(os.getcwd(), '/home/junjzhan/zhao_jg/dataset/data')
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
    return data, labels


def applyPCA(X, numComponents, windowSize):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]), dtype=np.float32)
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]), dtype=np.float32)
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels

def ind2Cubes_lab(X, gt, index, windowSize):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # get cube through index
    ind2cube = {}
    ind2label = {}
    patchesData = np.zeros((len(index), windowSize, windowSize, X.shape[2]), dtype=np.float32)
    patchesLabels = np.zeros((len(index)), dtype=np.float32)
    for i in range(len(index)):
        x, y = index[i]
        patch = zeroPaddedX[x: x + 2 * margin+1, y: y + 2 * margin+1]
        label = gt[x, y] - 1
        patchesData[i, :, :, :] = patch
        patchesLabels[i] = gt[x, y] - 1
        ind2cube.update({(x, y): patch})
        ind2label.update({(x, y): label})
    # ind2cube = dict(zip(index, patchesData))
    # ind2label = dict(zip(index, patchesLabels))
    return patchesData, patchesLabels

def indexToCube(final_pse_index, X):
    margin = int((13 - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    patchesData = []
    #patchesData = np.zeros((len(final_pse_index), 13, 13, X.shape[2]), dtype=np.float32)
    for i in range(len(final_pse_index)):
        x, y = final_pse_index[i]
        patch = zeroPaddedX[x: x + 2 * margin + 1, y: y + 2 * margin + 1]
        #patchesData[i, :, :, :] = patch
        patchesData.append(patch)
    return patchesData

def PerClassSplit(dataset, X, y, perclass, stratify, randomState=345):
    np.random.seed(randomState)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    train_all_index = []
    test_all_index = []
    # split data for indian pianes
    if (("IP" is dataset) & (perclass > 20)):
        for label in stratify:
            indexList = [i for i in range(len(y)) if y[i] == label]
            if (len(indexList) < 30):
                train_index = np.random.choice(indexList, 15, replace=True)
            else:
                train_index = np.random.choice(indexList,perclass, replace=True)
            for i in range(len(train_index)):
                index = train_index[i]
                X_train.append(X[index])
                y_train.append(label)
            test_index = [i for i in indexList if i not in train_index]

            for i in range(len(test_index)):
                index = test_index[i]
                X_test.append(X[index])
                y_test.append(label)

    else:
        for label in stratify:
            indexList = [i for i in range(len(y)) if y[i] == label]
            train_index = np.random.choice(indexList, perclass, replace=False)
            for i in range(len(train_index)):
                index = train_index[i]
                train_all_index.append(index)
                X_train.append(X[index])
                y_train.append(label)
            test_index = [i for i in indexList if i not in train_index]
            for i in range(len(test_index)):
                index = test_index[i]
                test_all_index.append(index)
                X_test.append(X[index])
                y_test.append(label)
            #train_all_index = np.append(train_all_index, train_index)
    return X_train, X_test, y_train, y_test, train_all_index, test_all_index


class MYDataset(torch.utils.data.Dataset):  # 需要继承data.Dataset
    def __init__(self, Datapath, Labelpath, transform):
        # 1. Initialize file path or list of file names.
        self.Datalist = np.load(Datapath)
        self.Labellist = (np.load(Labelpath)).astype(int)
        self.transform = transform

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        # 这里需要注意的是，第一步：read one data，是一个data
        index = index
        Data = self.transform(self.Datalist[index].astype('float64'))
        Data = Data.view(1, Data.shape[0], Data.shape[1], Data.shape[2])
        return Data, self.Labellist[index]

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.Datalist)


def feature_normalize(data):
    # data_shape = np.shape(data)
    # data = data.reshape(-1, data_shape[-1])
    # mu = np.mean(data, axis=0)
    # std = np.std(data, axis=0)
    # data = truediv((data - mu), std)
    # return data.reshape(data_shape)
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return truediv((data - mu), std)

# get random fs train sample index
def random_index(gt_map, perclass_num):
    nonzero = np.nonzero(gt_map)
    all_index = list(zip(*nonzero))  # (10249)
    per_class_index = {}
    train_index_list = {}
    train_index = []
    for num_class in range(np.max(gt_map)):  # 16个类别
        for j, (x, y) in enumerate(all_index):  # 遍历每个sample
            if gt_map[x, y] == num_class + 1:
                per_class_index.setdefault('class {}'.format(num_class), []).append((x, y))  # {dict:16} 每个类别的(x,y)坐标
        #  随机取 few shot 个sample
        class_list = list(per_class_index.values())[0][:]  # 强制转化为list
        per_class_index.clear()  # 清空dict
        few_shot_list = random.sample(class_list, perclass_num)  # 每个class list 随机选 few shot 个样本
        train_index_list.update(({'class {}'.format(num_class): few_shot_list}))
        for i in range(len(few_shot_list)):
            train_index.append(few_shot_list[i])
            # test index
            all_index.remove(few_shot_list[i])
    all_sample_index = list(zip(*nonzero))
    return all_sample_index, train_index, all_index

def get_XY(gt_map, train_index, test_index, ind2cubes):
    Ytrain = []
    Ytest = []
    Xtrain = []
    Xtest = []
    for i in range(len(train_index)):
        for key, value in ind2cubes.items():
            if key == train_index[i]:
                Ytrain.append(gt_map[train_index[i]].astype(int) - 1)
                Xtrain.append(value)
    for i in range(len(test_index)):
        for key, value in ind2cubes.items():
            if key == test_index[i]:
                Ytest.append(gt_map[test_index[i]].astype(int) - 1)
                Xtest.append(value)
    return Ytrain, Xtrain, Ytest, Xtest

def indexToCube_label(index, X, gt_map, windowSize):
    margin = int((windowSize - 1) / 2)
    X = padWithZeros(X, margin=margin)
    patches = []
    labels = []
    # patchesData = np.zeros((len(final_pse_index), 13, 13, X.shape[2]), dtype=np.float32)
    for i in range(len(index)):
        x, y = index[i]
        patch = X[x: x + 2 * margin + 1, y: y + 2 * margin + 1]
        label = gt_map[x, y] - 1
        # patchesData[i, :, :, :] = patch
        patches.append(patch)
        labels.append(label)
    return patches, labels

def gtAndneighbor_index(train_index):
    # step 1 get train GT (x,y)
    train_index = train_index.tolist()

    # step 2 get train GT mask (x,y) with stride
    mask_index = []
    for i in range(len(train_index)):
        x, y = train_index[i][0], train_index[i][1]
        per_mask_index = [(x - 1, y - 1), (x, y - 1), (x + 1, y-1), (x - 1, y),
                          (x + 1, y), (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]
        mask_index += per_mask_index

    # step 3 predict score > 0.8 (x,y)
    pre_thres_index = np.load('./Split_Data/pseudo_thres_index.npy').tolist()
    thres_index = []
    for i in range(len(pre_thres_index)):
        thres_index += [tuple(pre_thres_index[i])]

    # step 4 get common index between mask index and pre_thres_index
    pse_index = (list(set(mask_index) & set(thres_index)))
    np.save('./Split_Data/gt_neighbor_index.npy', pse_index)

    # step 5 add one train index
    #np.save('./Split_Data/gt+gt_index.npy', pse_index)# fs train index and pseudo index
    return pse_index


def ext_feature(ext_feature_net, Datapath, Labelpath, trans, batch_size):
    classi_data = MYDataset(Datapath, Labelpath, trans)  # supervised part
    classi_loader = torch.utils.data.DataLoader(dataset=classi_data, batch_size=batch_size, shuffle=True)
    for i, (data, label) in enumerate(classi_loader):  # supervised part
        data = data.cuda().float()  # tensor
        if i == 0:
            output_feature = torch.zeros(batch_size, 64).cuda().float()
            output_feature += ext_feature_net(data)
        else:
            output_feature = torch.cat((output_feature, ext_feature_net(data)), 0)
    return output_feature