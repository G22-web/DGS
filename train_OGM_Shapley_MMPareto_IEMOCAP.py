import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import math
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from model_OGM_Shapley_MMPareto import MaskedNLLLoss, LSTMModel, GRUModel, Model, MaskedMSELoss, FocalLoss
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
from torch.nn.parallel import DistributedDataParallel as DDP
from min_norm_solvers import MinNormSolver
import pandas as pd
import pickle as pk
import datetime
import ipdb
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.distributed as dist
import re
torch.cuda.empty_cache()

seed = 1475 # We use seed = 1475 on IEMOCAP and seed = 67137 on MELD
def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def _init_fn(worker_id):
        np.random.seed(int(seed)+worker_id)

def get_train_valid_sampler(trainset, valid=0.1, dataset='IEMOCAP'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset('MELD_features/MELD_features_raw1.pkl')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              shuffle=True)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              shuffle=True)

    testset = MELDDataset('MELD_features/MELD_features_raw1.pkl', train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                            shuffle=True)

    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory, worker_init_fn=_init_fn,
                              shuffle=True)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              shuffle=True)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory, worker_init_fn=_init_fn,
                             shuffle=True)

    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    """
    
    """
    losses, preds, labels, masks = [], [], [], []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    max_sequence_len = []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything()
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]        

        max_sequence_len.append(textf.size(0))
        
        log_prob, alpha, alpha_f, alpha_b, _ = model(textf, qmask, umask)
        lp_ = log_prob.transpose(0,1).contiguous().view(-1, log_prob.size()[2])
        labels_ = label.view(-1)
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_,1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'),[]

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted')*100, 2)
    
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, [alphas, alphas_f, alphas_b, vids]

def register_gradient_hooks(a, v, l, modality_weights):
    hooks = []
    def create_hook(feature, weight, name):
        def hook_fn(grad):
            with torch.no_grad():
                weight = weight.to(grad.device)  
                if args.modulation == 'OGM_GE':  # bug fixed
                    adjusted_grad = grad * weight + torch.zeros_like(grad).normal_(0,grad.std().item() + 1e-8)
                elif args.modulation == 'OGM':
                    adjusted_grad = grad * weight
                elif args.modulation == 'Normal':
                    # no modulation, regular optimization
                    adjusted_grad = grad
                # print(f"Gradient for {name} modality: {adjusted_grad}")
                return adjusted_grad
        return feature.register_hook(hook_fn)
    hooks.append(create_hook(a, modality_weights[0], "audio"))
    hooks.append(create_hook(v, modality_weights[1], "visual"))
    hooks.append(create_hook(l, modality_weights[2], "language"))

    return hooks

def calculate_discrepancy_ratio_fusion(f_t, f_a, f_v, f_fusion):
    d_t = torch.norm(f_t - f_fusion, p=2)  
    d_a = torch.norm(f_a - f_fusion, p=2)  
    d_v = torch.norm(f_v - f_fusion, p=2)  

    rho_t = d_t / (d_t + d_a + d_v) 
    rho_a = d_a / (d_t + d_a + d_v) 
    rho_v = d_v / (d_t + d_a + d_v)  

    return rho_t, rho_a, rho_v


def train_or_eval_graph_model(model, loss_function, dataloader, epoch, cuda, modals, scheduler=None, optimizer=None,
                              train=False, dataset='IEMOCAP'):
    # To be announced later



if __name__ == '__main__':
    path = './saved/IEMOCAP/'

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--base-model', default='LSTM', help='base recurrent model, must be one of DialogRNN/LSTM/GRU')

    parser.add_argument('--graph-model', action='store_true', default=True,
                        help='whether to use graph model after recurrent encoding')

    parser.add_argument('--nodal-attention', action='store_true', default=False,
                        help='True, whether to use nodal attention in graph model: Equation 4,5,6 in Paper')

    parser.add_argument('--windowp', type=int, default=10,
                        help='no use. context window size for constructing edges in graph model for past utterances')

    parser.add_argument('--windowf', type=int, default=10,
                        help='no use. context window size for constructing edges in graph model for future utterances')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')

    parser.add_argument('--l2', type=float, default=0.000003, metavar='L2', help='0.00003, L2 regularization weight')

    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')

    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='0.5, dropout rate')

    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')

    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')

    parser.add_argument('--class-weight', action='store_true', default=True, help='True，use class weights')

    parser.add_argument('--active-listener', action='store_true', default=False, help='no use. active listener')

    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')

    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    parser.add_argument('--graph_type', default='relation', help='relation/GCN3/DeepGCN/MMGCN/MMGCN2')

    parser.add_argument('--use_topic', action='store_true', default=False, help='False, whether to use topic information')

    parser.add_argument('--alpha', type=float, default=0.8, help='0.2, alpha')

    parser.add_argument('--alpha_', type=float, default=0.2, help='0.2， alpha')

    parser.add_argument('--multiheads', type=int, default=6, help='multiheads')

    parser.add_argument('--graph_construct', default='full', help='single/window/fc for MMGCN2; direct/full for others')

    parser.add_argument('--use_gcn', action='store_true', default=True,
                        help='False, whether to combine spectral and none-spectral methods or not')

    parser.add_argument('--use_residue', action='store_true', default=False,
                        help='False, whether to use residue information or not')

    parser.add_argument('--multi_modal', action='store_true', default=False,
                        help='whether to use multimodal information')

    parser.add_argument('--mm_fusion_mthd', default='concat',
                        help='method to use multimodal information: concat, gated, concat_subsequently')

    parser.add_argument('--modals', default='avl', help='modals to fusion')

    parser.add_argument('--av_using_lstm', action='store_true', default=False,
                        help='False, whether to use lstm in acoustic and visual modality')

    parser.add_argument('--Deep_GCN_nlayers', type=int, default=4, help='Deep_GCN_nlayers')

    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')

    parser.add_argument('--use_speaker', action='store_true', default=True, help='True, whether to use speaker embedding')

    parser.add_argument('--use_modal', action='store_true', default=True, help='False,whether to use modal embedding')

    parser.add_argument('--norm', default='LN2', help='NORM type')

    parser.add_argument('--testing', action='store_true', default=False, help='testing')

    parser.add_argument('--num_L', type=int, default=3, help='num_hyperconvs')

    parser.add_argument('--num_K', type=int, default=4, help='num_convs')

    parser.add_argument('--fusion_method', default='concat', type=str, choices=['sum', 'concat', 'gated', 'film'])

    parser.add_argument('--modulation', default='OGM_GE', type=str, choices=['Normal', 'OGM', 'OGM_GE'])

    # parser.add_argument('--gpu_ids', default='3', type=str, help='GPU ids')

    parser.add_argument('--lr_decay_step', default=20, type=int, help='20,where learning rate decays')

    parser.add_argument('--lr_decay_ratio', default=1, type=float, help='0.1,decay coefficient')

    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')

    parser.add_argument('--modulation_ends', default=50, type=int, help='where modulation ends')

    args = parser.parse_args()
    today = datetime.datetime.now()
    print(args)

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    # gpu_ids = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0')

    if args.av_using_lstm:
        name_ = args.mm_fusion_mthd+'_'+args.modals+'_'+args.graph_type+'_'+args.graph_construct+'using_lstm_'+args.Dataset
    else:
        name_ = args.mm_fusion_mthd+'_'+args.modals+'_'+args.graph_type+'_'+args.graph_construct+str(args.Deep_GCN_nlayers)+'_'+args.Dataset

    if args.use_speaker:
        name_ = name_+'_speaker'
    if args.use_modal:
        name_ = name_+'_modal'

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    cuda       = args.cuda
    n_epochs   = args.epochs
    batch_size = args.batch_size
    modals = args.modals
    feat2dim = {'IS10':1582,'3DCNN':512,'textCNN':100,'bert':768,'denseface':342,'MELD_text':600,'MELD_audio':300}
    D_audio = feat2dim['IS10'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024 #feat2dim['textCNN'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_text']

    if args.multi_modal:
        if args.mm_fusion_mthd=='concat':
            if modals == 'avl':
                D_m = D_audio+D_visual+D_text
            elif modals == 'av':
                D_m = D_audio+D_visual
            elif modals == 'al':
                D_m = D_audio+D_text
            elif modals == 'vl':
                D_m = D_visual+D_text
            else:
                raise NotImplementedError
        else:
            D_m = 1024
    else:
        if modals == 'a':
            D_m = D_audio
        elif modals == 'v':
            D_m = D_visual
        elif modals == 'l':
            D_m = D_text
        else:
            raise NotImplementedError
    D_g = 1024 if args.Dataset=='IEMOCAP' else 1024
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100
    graph_h = 512
    n_speakers = 9 if args.Dataset=='MELD' else 2
    n_classes  = 7 if args.Dataset=='MELD' else 6 if args.Dataset=='IEMOCAP' else 1

    if args.graph_model:
        seed_everything()

        model = Model(args.base_model,
                                 D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                                 fusion_type=args.fusion_method,
                                 n_speakers=n_speakers,
                                 max_seq_len=200,
                                 window_past=args.windowp,
                                 window_future=args.windowf,
                                 n_classes=n_classes,
                                 listener_state=args.active_listener,
                                 context_attention=args.attention,
                                 dropout=args.dropout,
                                 nodal_attention=args.nodal_attention,
                                 no_cuda=args.no_cuda,
                                 graph_type=args.graph_type,
                                 use_topic=args.use_topic,
                                 alpha=args.alpha,
                                 multiheads=args.multiheads,
                                 graph_construct=args.graph_construct,
                                 use_GCN=args.use_gcn,
                                 use_residue=args.use_residue,
                                 D_m_v = D_visual,
                                 D_m_a = D_audio,
                                 modals=args.modals,
                                 att_type=args.mm_fusion_mthd,
                                 av_using_lstm=args.av_using_lstm,
                                 Deep_GCN_nlayers=args.Deep_GCN_nlayers,
                                 dataset=args.Dataset,
                                 use_speaker=args.use_speaker,
                                 use_modal=args.use_modal,
                                 norm = args.norm,
                                 num_L = args.num_L,
                                 num_K = args.num_K)
        model.apply(weight_init)
        model.to(device)
        # model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        print ('Graph NN with', args.base_model, 'as base model.')
        name = 'Graph'

    else:
        if args.base_model == 'GRU':
            model = GRUModel(D_m, D_e, D_h, 
                              n_classes=n_classes, 
                              dropout=args.dropout)

            print ('Basic GRU Model.')


        elif args.base_model == 'LSTM':
            model = LSTMModel(D_m, D_e, D_h, 
                              n_classes=n_classes, 
                              dropout=args.dropout)

            print ('Basic LSTM Model.')

        else:
            print ('Base model must be one of DialogRNN/LSTM/GRU/Transformer')
            raise NotImplementedError

        name = 'Base'

    if cuda:
        model.cuda()

    if args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668])

    if args.Dataset == 'MELD':
        loss_function = FocalLoss()
    else:
        if args.class_weight:
            if args.graph_model:
                #loss_function = FocalLoss()
                loss_function  = nn.NLLLoss(loss_weights.cuda() if cuda else loss_weights)
            else:
                loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        else:
            if args.graph_model:
                loss_function = nn.NLLLoss()
            else:
                loss_function = MaskedNLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.l2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    lr = args.lr
    
    if args.Dataset == 'MELD':
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                    batch_size=batch_size,
                                                                    num_workers=0)
    elif args.Dataset == 'IEMOCAP':
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=0)
    else:
        print("There is no such dataset")

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_micro_fscore, all_macro_fscore, all_acc, all_loss = [], [], [], [], []

    if args.testing:
        state = torch.load("best_model.pth.tar")
        model.load_state_dict(state)
        print('testing loaded model')
        test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_graph_model(model, loss_function, test_loader, 0, cuda, args.modals, dataset=args.Dataset)
        print('test_acc:',test_acc,'test_fscore:',test_fscore)

    for e in range(n_epochs):
        start_time = time.time()

        if args.graph_model:
            train_loss, train_loss_a, train_loss_v, train_loss_l, train_acc, _, _, train_fscore, train_avg_micro_fscore, train_avg_macro_fscore, _, _, _, _, _ = train_or_eval_graph_model(
                model, loss_function, train_loader, e, cuda, args.modals, scheduler, optimizer, True, dataset=args.Dataset)
            valid_loss, valid_loss_a, valid_loss_v, valid_loss_l, valid_acc, _, _, valid_fscore, valid_avg_micro_fscore, valid_avg_macro_fscore, _, _, _, _, _ = train_or_eval_graph_model(
                model, loss_function, valid_loader, e, cuda, args.modals, dataset=args.Dataset)
            test_loss, test_loss_a, test_loss_v, test_loss_l, test_acc, test_label, test_pred, test_fscore, test_avg_micro_fscore, test_avg_macro_fscore, _, _, _, _, _ = train_or_eval_graph_model(
                model, loss_function, test_loader, e, cuda, args.modals, dataset=args.Dataset)
            all_fscore.append(test_fscore)
            all_micro_fscore.append(test_avg_micro_fscore)
            all_macro_fscore.append(test_avg_macro_fscore)
            all_acc.append(test_acc)


        else:
            train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(model, loss_function, train_loader, e, optimizer, True)
            valid_loss, valid_acc, _, _, _, valid_fscore, _ = train_or_eval_model(model, loss_function, valid_loader, e)
            test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(model, loss_function, test_loader, e)
            all_fscore.append(test_fscore)

        if best_loss == None or best_loss > test_loss:
            best_loss, best_label, best_pred = test_loss, test_label, test_pred

        if best_fscore == None or best_fscore < test_fscore:
            best_fscore = test_fscore
            best_label, best_pred = test_label, test_pred
            #test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_graph_model(model, loss_function, test_loader, e, cuda, args.modals, dataset=args.Dataset)

        if args.tensorboard:
            writer.add_scalar('test: accuracy', test_acc, e)
            writer.add_scalar('test: fscore', test_fscore, e)
            writer.add_scalar('train: accuracy', train_acc, e)
            writer.add_scalar('train: fscore', train_fscore, e)

        print(
            'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, test_loss: {}, test_loss_a: {}, test_loss_v: {}, test_loss_l: {},test_acc: {}, test_fscore: {}, time: {} sec'. \
            format(e + 1, train_loss, train_acc, train_fscore, test_loss, test_loss_a, test_loss_v, test_loss_l, test_acc, test_fscore,
                   round(time.time() - start_time, 2)))
        if (e + 1) % 10 == 0:
            print('best Weight-F-Score:', max(all_fscore))
            print('best Miceo-F-Score:', max(all_micro_fscore))
            print('best Macro-F-Score:', max(all_macro_fscore))
            print('best Acc-Score:', max(all_acc))
            print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
            print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))

        
    

    if args.tensorboard:
        writer.close()
    if not args.testing:
        print('Test performance..')
        print('Wetight-F-Score:', max(all_fscore))
        print('Miceo-F-Score:', max(all_micro_fscore))
        print('Macro-F-Score:', max(all_macro_fscore))
        print('Acc-Score:', max(all_acc)) 
        if not os.path.exists("record_{}_{}_{}.pk".format(today.year, today.month, today.day)):
            with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day),'wb') as f:
                pk.dump({}, f)
        with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'rb') as f:
            record = pk.load(f)
        key_ = name_
        if record.get(key_, False):
            record[key_].append(max(all_fscore))
        else:
            record[key_] = [max(all_fscore)]
        if record.get(key_+'record', False):
            record[key_+'record'].append(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
        else:
            record[key_+'record'] = [classification_report(best_label, best_pred, sample_weight=best_mask,digits=4)]
        with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day),'wb') as f:
            pk.dump(record, f)

        print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
        print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))
