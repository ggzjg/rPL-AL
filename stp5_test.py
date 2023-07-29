import numpy as np

import stp6_get_cls_map
import net
import torch
from torchvision import transforms
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import time
import argparse
from createData import MYDataset
from torchsummary import summary
import torch.nn.functional as F
import scipy.io as sio

def predict(model, Datapath, Labelpath):
    model.eval()
    model = model.cuda()
    test_data = MYDataset(Datapath, Labelpath, trans)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
    prediction = []
    extract_feat = []
    for data, label in test_loader:
        data = data.cuda().float()
        feat, out = model(data)
        for num in range(len(out)):
            prediction.append(np.array(out[num].cpu().detach().numpy()))
            extract_feat.append(np.array(feat[num].cpu().detach().numpy()))
    pred_label = np.argmax(np.array(prediction), axis=1)
    return prediction, extract_feat, pred_label


def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def result_reports(y_pred, Labelpath, name, iter):
    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
            , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                        'Fallow_smooth',
                        'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                        'Vinyard_untrained', 'Vinyard_vertical_trellis']
    elif name == 'PU':
        target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
    Label = np.load(Labelpath).astype(int)
    classification = classification_report(Label, y_pred, digits=4, target_names=target_names)
    oa = accuracy_score(Label, y_pred) * 100
    confusion = confusion_matrix(Label, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    each_acc = each_acc * 100
    aa = aa * 100
    kappa = cohen_kappa_score(Label, y_pred) * 100
    print('========== After iter={} Pseudo Labeling =========='.format(iter))
    print('OA:', oa)
    print('AA:', aa)
    print('Kappa:', kappa)
    return classification, confusion, oa, each_acc, aa, kappa,


if __name__ == '__main__':
    dataset_names = ['IP', 'SA', 'PU']
    parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                     " various hyperspectral datasets")
    parser.add_argument('--dataset', type=str, default='IP', choices=dataset_names,
                        help="Dataset to use.")
    parser.add_argument('--perclass', type=int, default=5)  # few-shot number
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--windowSize', type=int, default=13)
    parser.add_argument('--numComponents', type=int, default=30)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--load_model_path', type=str, default='./Params/connect_area_train_fs5_iter1.pth')
    parser.add_argument('--iter_idx', type=str, default='1')
    parser.add_argument('--epo', type=str, default='1')
    args = parser.parse_args()
    epoch, dataset, windowSize, numComponents = args.epoch, args.dataset, \
                                                args.windowSize, args.numComponents
    model = net.SSFTTnet().cuda()
    model.load_state_dict(torch.load(args.load_model_path))
    trans = transforms.Compose(transforms=[
            transforms.ToTensor(),
            transforms.Normalize(np.zeros(numComponents), np.ones(numComponents))])
    # test
    Datapath = './Split_Data/{}_{}_shot_test_data.npy'.format(args.dataset, args.perclass)
    Labelpath = './Split_Data/{}_{}_shot_test_label.npy'.format(args.dataset, args.perclass)
    pred_score, extract_feat, pred_label = predict(model, Datapath, Labelpath)
    classification, confusion, oa, each_acc, aa, kappa,  = result_reports(pred_label, Labelpath, dataset, args.iter_idx)
    classification = str(classification)
    file_name = "./Metric/epo_{}_IP_test.txt".format(args.epo)
    with open(file_name, 'a') as x_file:
        x_file.write('||||||||||||||||||||||| epo {} ||||||||||||||||||||||||||||||||\n'.format(args.epo))
        x_file.write('epo {} iter index {}th start\n'.format(args.epo, args.iter_idx))
        x_file.write(' Overall accuracy (%):{}'.format(oa))
        x_file.write('\n')
        x_file.write(' Average accuracy (%):{}'.format(aa))
        x_file.write('\n')
        x_file.write(' Kappa accuracy (%):{}'.format(kappa))
        x_file.write('\n')
        x_file.write('---------------------------------------------------\n')
        x_file.write(' Each accuracy (%):{} '.format(each_acc))
        x_file.write('\n')
        x_file.write('---------------------------------------------------\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('---------------------------------------------------\n')
        x_file.write('{}'.format(confusion))
        x_file.write('---------------------------------------------------\n')
        x_file.write('epo {} iter index {}th done \n'.format(args.epo, args.iter_idx))
        x_file.write('\n')
        x_file.write('\n')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    X_pca, y_all = createImageCubes(X_pca, y, windowSize=windowSize)
    stp6_get_cls_map.get_cls_map.(model, device, all_data_loader, y_all)