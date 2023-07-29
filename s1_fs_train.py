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
        # print(
        #     'Train Loss: {:.6f}, Acc: {:.6f}'.format(classi_loss / (len(classi_data)),
        #                                              train_acc / (len(classi_data)),
        #                                              ))
        # can change to more better , add trian loss to get best model
        if (train_acc / (len(classi_data)) >= best_acc) and (
                (classi_loss / (len(classi_data))) < best_loss):
            best_model_wts = copy.deepcopy(encoder.state_dict())
            best_acc = train_acc / (len(classi_data))
            best_loss = (classi_loss / (len(classi_data)))
    torch.save(best_model_wts, './Params/fs_train_fs'+str(args.perclass)+'.pth')
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
    all_index, train_index, test_index = random_index(gt_map, perclass)
    train_data, train_label = indexToCube_label(train_index, X, gt_map, windowSize)
    test_data, test_label = indexToCube_label(test_index, X, gt_map, windowSize)
    np.save('./Split_Data/{}_{}_shot_train_index.npy'.format(args.dataset, args.perclass), train_index)
    np.save('./Split_Data/{}_{}_shot_test_index.npy'.format(args.dataset, args.perclass), test_index)
    np.save('./Split_Data/{}_{}_shot_train_data.npy'.format(args.dataset, args.perclass), train_data)
    np.save('./Split_Data/{}_{}_shot_train_label.npy'.format(args.dataset, args.perclass), train_label)
    np.save('./Split_Data/{}_{}_shot_test_data.npy'.format(args.dataset, args.perclass), test_data)
    np.save('./Split_Data/{}_{}_shot_test_label.npy'.format(args.dataset, args.perclass), test_label)

    # sio.savemat('./Split_Data/{}_{}_shot_train_index.mat'.format(args.dataset, args.perclass),
    #             {'{}_{}_shot_train_index.mat'.format(args.dataset, args.perclass):train_index})
    #
    # sio.savemat('./Split_Data/{}_{}_shot_test_index.mat'.format(args.dataset, args.perclass),
    #             {'{}_{}_shot_test_index.mat'.format(args.dataset, args.perclass): test_index})

    Datapath = './Split_Data/{}_{}_shot_train_data.npy'.format(args.dataset, args.perclass)
    Labelpath = './Split_Data/{}_{}_shot_train_label.npy'.format(args.dataset, args.perclass)
    trans = transforms.Compose(transforms=[
        transforms.ToTensor(),
        transforms.Normalize(np.zeros(numComponents), np.ones(numComponents))])
    model = net.SSFTTnet().cuda()
    train_feature = train(model, Datapath, Labelpath, trans, epochs=epoch)
    #print('few shot train finished-----------------------------')
    # sio.savemat('./Feature/fs{}_train_data.mat'.format(perclass),
    #             {'fs{}_train_data'.format(perclass): np.asarray(train_data)[:, 7, 7, :]})
    # sio.savemat('./Feature/fs{}_test_data.mat'.format(perclass),
    #             {'fs{}_test_data'.format(perclass): np.asarray(test_data)[:, 7, 7, :]})
    # sio.savemat('./Feature/fs{}_train_label.mat'.format(perclass),
    #             {'fs{}_train_label'.format(perclass): train_label})
    # sio.savemat('./Feature/fs{}_test_label.mat'.format(perclass),
    #             {'fs{}_test_label'.format(perclass): test_label})
    # sio.savemat('./Feature/fs{}_train_index.mat'.format(perclass),
    #             {'fs{}_train_index'.format(perclass): train_index})
    # sio.savemat('./Feature/fs{}_test_index.mat'.format(perclass),
    #             {'fs{}_test_index'.format(perclass): test_index})




