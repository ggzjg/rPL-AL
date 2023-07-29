import createData
import net
import torch
from torchvision import transforms
import copy
import numpy as np
import argparse
import scipy.io as sio
from createData import applyPCA, random_index, feature_normalize, indexToCube_label
def train(encoder, Datapath, Labelpath, trans, epochs):
    classi_data = createData.MYDataset(Datapath, Labelpath, trans)  # supervised part
    classi_loader = torch.utils.data.DataLoader(dataset=classi_data, batch_size=args.batch_size, shuffle=True)
    optim_classi = torch.optim.Adam(encoder.parameters(), lr=args.classi_lr, weight_decay=0.00005)
    criterion = torch.nn.CrossEntropyLoss()
    best_loss = 10000
    best_model_wts = copy.deepcopy(encoder.state_dict())
    best_acc = 0
    for epoch in range(epochs):
        train_acc = 0
        epoch_classiloss = 0
        # print(" Epoch No  {} ".format(epoch + 1))
        for i, (data, label) in enumerate(classi_loader):  # supervised part
            data = data.cuda().float()  # tensor:(128,1,30,13,13)
            label = label.cuda()
            feat, z = encoder(data)
            classi_loss = criterion(z, label)
            epoch_classiloss += classi_loss.item()
            pred = torch.max(z, 1)[1]
            train_correct = (pred == label).sum()
            train_acc += train_correct.item()
            optim_classi.zero_grad()
            classi_loss.backward(retain_graph=True)
            optim_classi.step()
        # print('Train Loss: {:.6f}, Acc: {:.6f}'.format(classi_loss / (len(classi_data)),
        #                                              train_acc / (len(classi_data)),))
        if (train_acc / (len(classi_data)) >= best_acc) and (
                (classi_loss / (len(classi_data))) < best_loss):
            best_model_wts = copy.deepcopy(encoder.state_dict())
            best_acc = train_acc / (len(classi_data))
            best_loss = (classi_loss / (len(classi_data)))
    torch.save(best_model_wts, args.save_model_path)
    return feat


if __name__ == '__main__':
    dataset_names = ['IP', 'SA', 'PU']
    parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                                 " various hyperspectral datasets")
    parser.add_argument('--dataset', type=str, default='IP', choices=dataset_names,
                        help="Dataset to use.")
    parser.add_argument('--perclass', type=int, default=5)  # few-shot number
    parser.add_argument('--device', type=str, default="cuda:0", choices=("cuda:0", "cuda:1"))
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--windowSize', type=int, default=13)
    parser.add_argument('--numComponents', type=int, default=30)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--classi_lr', type=float, default=1e-3)
    parser.add_argument('--train', type=bool, default=1)
    parser.add_argument('--save_model_path', type=str, default='./Params/connect_area_train_fs5_iter1.pth')
    parser.add_argument('--load_model_path', type=str, default='./Params/fs_train_fs5.pth')
    parser.add_argument('--iter_idx', type=str, default='1')
    parser.add_argument('--epo', type=str, default='1')
    args = parser.parse_args()
    epoch = args.epoch
    dataset = args.dataset
    perclass = args.perclass  # few-shot
    windowSize = args.windowSize  # patch_size
    numComponents = args.numComponents  # 降维后的通道数
    output_units = 9 if (dataset == 'PU' or dataset == 'PC') else 16  # 类别数目
    stratify = np.arange(0, output_units, 1)
    hsi_data, gt_map = createData.loadData(dataset)  # 145*145*200, 145*145
    X = createData.applyPCA(hsi_data, numComponents, windowSize)
    X = createData.feature_normalize(X)

    connect_pseudo_label_map = np.load('./Split_Data/final_connect_pseudolabel_fs'+str(args.perclass)+'_iter'+args.iter_idx+'.npy',allow_pickle=True)
    # pseudo_ind2label = np.load('./Split_Data/pseudo_ind2label_fs'+str(args.perclass)+'_iter'+args.iter_idx+'.npy').tolist()
    #
    # shot_train_index = np.load('./Split_Data/{}_{}_shot_train_index.npy'.format(args.dataset, args.perclass)).tolist()
    # # add train gt label
    # shot_train_label = np.load('./Split_Data/{}_{}_shot_train_label.npy'.format(args.dataset, args.perclass)).tolist()
    # for i in range(len(shot_train_label)):
    #     pseudo_ind2label.update({(shot_train_index[i][0], shot_train_index[i][1]): np.array(shot_train_label[i])})
    #
    final_connect_index = np.load('./Split_Data/final_connect_index_fs'+str(args.perclass)+'_iter'+args.iter_idx+'.npy',allow_pickle=True).tolist()
    final_connect_data, final_connect_GTlabel = indexToCube_label(final_connect_index, X, gt_map, windowSize)
    final_connect_label = []
    for i in range(len(final_connect_index)):
        final_connect_label.append(connect_pseudo_label_map[(final_connect_index[i][0],
                                                    final_connect_index[i][1])]-1)

    np.save('./Split_Data/final_connect_data_fs'+str(args.perclass)+'_iter'+args.iter_idx+'.npy', final_connect_data)
    np.save('./Split_Data/final_connect_label_fs'+str(args.perclass)+'_iter'+args.iter_idx+'.npy', final_connect_label)
    Datapath = './Split_Data/final_connect_data_fs'+str(args.perclass)+'_iter'+args.iter_idx+'.npy'
    Labelpath = './Split_Data/final_connect_label_fs'+str(args.perclass)+'_iter'+args.iter_idx+'.npy'
    trans = transforms.Compose(transforms=[
        transforms.ToTensor(),
        transforms.Normalize(np.zeros(numComponents), np.ones(numComponents))])
    model = net.SSFTTnet().cuda()
    model.load_state_dict(torch.load(args.load_model_path))
    train_feature = train(model, Datapath, Labelpath, trans, epochs=epoch)
    # print('connect train finished----------------------')





