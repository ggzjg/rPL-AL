import numpy as np
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
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def result_reports(y_pred, Labelpath, name):
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
    print('========== FS Only ==========')
    print('OA:', oa)
    print('AA:', aa)
    print('Kappa:', kappa)
    return classification, confusion, oa, each_acc, aa, kappa,


if __name__ == '__main__':
    dataset_names = ['IP', 'SA', 'PU']
    parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                     " various hyperspectral datasets")
    parser.add_argument('--dataset', type=str, default='PU', choices=dataset_names,
                        help="Dataset to use.")
    parser.add_argument('--perclass', type=int, default=5)  # few-shot number
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--windowSize', type=int, default=13)
    parser.add_argument('--numComponents', type=int, default=30)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--load_model_path', type=str)
    parser.add_argument('--iter_idx', type=str)

    args = parser.parse_args()
    epoch, dataset, windowSize, numComponents, perclass= args.epoch, args.dataset, \
                                                args.windowSize, args.numComponents, args.perclass
    model = net.SSFTTnet().cuda()
    model.load_state_dict(torch.load(args.load_model_path))
    trans = transforms.Compose(transforms=[
            transforms.ToTensor(),
            transforms.Normalize(np.zeros(numComponents), np.ones(numComponents))])
    # test
    Datapath = './Split_Data/{}_{}_shot_test_data.npy'.format(args.dataset, args.perclass)
    Labelpath = './Split_Data/{}_{}_shot_test_label.npy'.format(args.dataset, args.perclass)
    pred_score, test_feat, pred_label = predict(model, Datapath, Labelpath)
    test_feat = np.asarray(test_feat)
    # sio.savemat('./Feature/fs{}_test_feature.mat'.format(perclass),
    #             {'fs{}_test_feature'.format(perclass): test_feat})

    #sio.savemat('./Feature/fs{}_pred_scores.mat'.format(args.perclass), {'pred_scores_arr': pred_scores_arr})
    if args.iter_idx == '1':
        classification, confusion, oa, each_acc, aa, kappa,  = result_reports(pred_label, Labelpath, dataset)


    # # pass train set to trained model
    # Datapath = './Split_Data/{}_{}_shot_train_data.npy'.format(args.dataset, args.perclass)
    # Labelpath = './Split_Data/{}_{}_shot_train_label.npy'.format(args.dataset, args.perclass)
    # _, train_feature, _ = predict(model, Datapath, Labelpath)
    # train_feature = np.asarray(train_feature)
    # sio.savemat('./Feature/fs{}_train_feature.mat'.format(perclass),
    #             {'fs{}_train_feature'.format(perclass): train_feature})


    #select > 0.9 threshold
    thres_score = F.softmax(torch.Tensor(pred_score), dim=1)
    pred_scores_arr = np.asarray(thres_score)
    #sio.savemat('./Feature/fs{}_pred_scores.mat'.format(args.perclass), {'pred_scores_arr': pred_scores_arr})
    #np.save('./Split_Data/fs{}_pred_scores.npy'.format(args.perclass), pred_scores_arr)
    test_index = np.load('./Split_Data/{}_{}_shot_test_index.npy'.format(args.dataset, args.perclass))  # 10169
    test_ind2pre_score = {}
    for i in range(len(thres_score)):
        tup = (test_index[i][0], test_index[i][1])
        v = thres_score[i]
        test_ind2pre_score.update({tup: v})  # [(x,y):predict_score]
    pseudo_ind2label = {}
    pseudo_ind2score = {}
    pseudo_thres_index = []
    score = []
    for key, value in test_ind2pre_score.items():
        if (max(value) > 0.9):
            # get key value
            pseudo_thres_index.append((key[0], key[1]))
            # all_index_label.update({key: np.argmax(value.cpu().numpy())})
            pseudo_ind2label.update({key: np.argmax(np.array(value), axis=0)})
            pseudo_ind2score.update({key: np.array(value)})
    #         score.append(value)
    # sio.savemat('./Feature/fs{}_pred_scores.mat'.format(args.perclass), np.asarray(score))
    # sio.savemat('./Feature/fs{}_thres_index.mat'.format(perclass),
    #             {'fs{}_thres_index'.format(perclass): train_feature})
    np.save('./Split_Data/pseudo_thres_index_fs'+str(args.perclass)+'_iter'+args.iter_idx+'.npy', pseudo_thres_index)
    np.save('./Split_Data/pseudo_ind2label_fs'+str(args.perclass)+'_iter'+args.iter_idx+'.npy', pseudo_ind2label)  # [(x,y)]
    np.save('./Split_Data/pseudo_ind2score_fs'+str(args.perclass)+'_iter'+args.iter_idx+'.npy', pseudo_ind2score)  # [(x,y): label]
