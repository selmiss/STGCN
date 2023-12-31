import logging
import os
import argparse
import math
import random
import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from script import dataloader, utility, earlystopping, gpruning
from scipy import sparse
from model import models


def set_env(seed):
    # Set available CUDA devices
    # This option is crucial for an multi-GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

ckp = "./checkpoints/gpstg-metr-la-30/0.2692_metr-la.pth"
ckp_save = "./checkpoints/loss_record"
if_train = True
if_load = True
def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--dataset', type=str, default='metr-la', choices=['metr-la', 'pems-bay', 'pemsd7-m'])
    parser.add_argument('--n_his', type=int, default=12)
    parser.add_argument('--tf_rate', type=int, default=100)
    parser.add_argument('--n_pred', type=int, default=6, help='the number of time interval for predcition, default as 3')
    parser.add_argument('--time_intvl', type=int, default=5)
    parser.add_argument('--Kt', type=int, default=3)
    parser.add_argument('--stblock_num', type=int, default=2)
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
    parser.add_argument('--Ks', type=int, default=3, choices=[3, 2])
    parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv', choices=['cheb_graph_conv', 'graph_conv'])
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap', choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.0005, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50, help='epochs, default as 10000')
    parser.add_argument('--opt', type=str, default='adam', help='optimizer, default as adam')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    # For stable experiment results
    set_env(args.seed)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num

    # blocks: settings of channel size in st_conv_blocks and output layer,
    # using the bottleneck design in st_conv_blocks
    blocks = []
    blocks.append([1])
    for l in range(args.stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([1])
    
    return args, device, blocks


def data_preparate(args, device):
    
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, args.dataset)
    data_col = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).shape[0]
    # recommended dataset split rate as train: val: test = 60: 20: 20, 70: 15: 15 or 80: 10: 10
    # using dataset split rate as train: val: test = 70: 15: 15
    val_and_test_rate = 0.15

    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)
    # train, val, test = dataloader.load_data(args.dataset, len_train, len_val)
    # # (23991, 207) raw data
    # train = train.iloc[:, :200]
    # val = val.iloc[:, :200]
    # test = test.iloc[:, :200]
        
    adj, n_vertex = dataloader.load_adj(args.dataset)
    img = adj.A
    # img = img[:200, :200]
    
    train, val, test, img = gpruning.pruning(args.dataset, img, 0, 200, len_train, len_val)
    gso_image = Image.fromarray(img * 255)
    gso_image.show()

    adj = sparse.csr_matrix(img)

    gso = utility.calc_gso(adj, args.gso_type)
    
    if args.graph_conv_type == 'cheb_graph_conv':
        gso = utility.calc_chebynet_gso(gso)
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)

    args.gso = torch.from_numpy(gso).to(device)
    
    

    zscore = preprocessing.StandardScaler()
    # (23991, 207) standard data
    train = zscore.fit_transform(train)
    val = zscore.transform(val)
    test = zscore.transform(test)
    
    train = train[:int(len(train) * args.tf_rate /100)]
    val = train[:int(len(val) * args.tf_rate /100)]


    x_train, y_train = dataloader.data_transform(train, args.n_his, args.n_pred, device)
    # print(x_train.shape, y_train.shape, y_train[0])
    # torch.Size([23976, 1, 12, 207]) torch.Size([23976, 207])
    x_val, y_val = dataloader.data_transform(val, args.n_his, args.n_pred, device)
    x_test, y_test = dataloader.data_transform(test, args.n_his, args.n_pred, device)

    train_data = utils.data.TensorDataset(x_train, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)
    val_data = utils.data.TensorDataset(x_val, y_val)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    return n_vertex, zscore, train_iter, val_iter, test_iter


def prepare_model(args, blocks, n_vertex, n_internal=200):
    loss = nn.MSELoss()
    es = earlystopping.EarlyStopping(mode='min', min_delta=0.0, patience=args.patience)

    if args.graph_conv_type == 'cheb_graph_conv':
        model = models.STGCNChebGraphConv(args, blocks, n_internal, n_vertex).to(device)
    else:
        model = models.STGCNGraphConv(args, blocks, n_internal, n_vertex).to(device)

    if args.opt == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    else:
        raise NotImplementedError(f'ERROR: The optimizer {args.opt} is not implemented.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    return loss, es, model, optimizer, scheduler


def train(loss, args, optimizer, scheduler, es, model, train_iter, val_iter, save_dir='./checkpoints'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    lost_least = 10
    train_loss_list = []
    val_loss_list = []
    for epoch in range(args.epochs):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        for x, y in tqdm.tqdm(train_iter):

            y_pred = model(x).view(len(x), -1)  # [batch_size, num_nodes]
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        scheduler.step()
        val_loss = val(model, val_iter)
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))
        train_loss_list.append(l_sum / n)
        val_loss_list.append(val_loss)
        if val_loss < lost_least:
            checkpoint_name = str(round(val_loss.item(),4)) + "_" + args.dataset + ".pth"
            lost_least = val_loss
            save_path = os.path.join(save_dir, checkpoint_name)
            # torch.save(model.state_dict(), save_path)
        
        if es.step(val_loss):
            np.save(os.path.join(save_dir, "train_loss.npy"), np.array(train_loss_list))
            np.save(os.path.join(save_dir, "val_loss.npy"), np.array(val_loss_list))
            print('Early stopping.')
            break
    torch.save(model.state_dict(), save_path)
    np.save(os.path.join(save_dir, "train_loss.npy"), np.array(train_loss_list))
    np.save(os.path.join(save_dir, "val_loss.npy"), np.array(val_loss_list))

@torch.no_grad()
def val(model, val_iter):
    model.eval()
    l_sum, n = 0.0, 0
    for x, y in val_iter:
        y_pred = model(x).view(len(x), -1)
        l = loss(y_pred, y)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    return torch.tensor(l_sum / n)


@torch.no_grad() 
def test(zscore, loss, model, test_iter, args):
    model.eval()
    test_MSE = utility.evaluate_model(model, loss, test_iter)
    test_MAE, test_RMSE, test_WMAPE = utility.evaluate_metric(model, test_iter, zscore)
    utility.evaluate_graph(model, test_iter, zscore)
    print(f'Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}')


def load_model_from_checkpoint(model, ckp_path):
    if os.path.exists(ckp_path):
        if torch.cuda.is_available():
            checkpoint = torch.load(ckp_path)
        else:
            checkpoint = torch.load(ckp_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        print('load model checkpoint from', ckp_path)
    else:
        print(ckp_path, 'file not exist')


if __name__ == "__main__":
    # Logging
    #logger = logging.getLogger('stgcn')
    #logging.basicConfig(filename='stgcn.log', level=logging.INFO)
    logging.basicConfig(level=logging.INFO)

    args, device, blocks = get_parameters()
    n_vertex, zscore, train_iter, val_iter, test_iter = data_preparate(args, device)
    loss, es, model, optimizer, scheduler = prepare_model(args, blocks, 207)
    if if_load:
        load_model_from_checkpoint(model, ckp)
    if if_train:
        train(loss, args, optimizer, scheduler, es, model, train_iter, val_iter, ckp_save)

    test(zscore, loss, model, test_iter, args)
