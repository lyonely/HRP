#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
import argparse
from quantization.sq.utils import quant_framework
import logging
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from quantization.epq.pqact import ActPQ
from quantization.utils import QParam, Layer_Qparam, sizeTracker, result_container, fetchAssign
from h2gc import h2gc_sparse
from benchmark import GCN, MLP
from datetime import datetime
import numpy as np
import pandas as pd
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def argParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gdc', action='store_true',
                        help='Use GDC preprocessing.')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--block_size', type=int, default=32)
    parser.add_argument('--ncents', type=int, default=64, 
                        help='the upper limit of learned clusters and the upper limit of learned clusters for each batch if mini_batch is true')
    parser.add_argument('--try_cluster', type=int, default=15,
                        help='number of attempts to find more centroids')
    parser.add_argument('--n_iter', type=int, default=10,
                        help='number of iteration for cluster')
    parser.add_argument('--mini_batch', action="store_true", 
                        help='apply batch method')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='number of nodes in each batch')
    parser.add_argument('--path', type=str, default=f'./pq_data/')
    parser.add_argument('--pqnt', action="store_true", #default=True,
                        help='apply EPQ on input data')
    parser.add_argument('--act_qnt', action="store_true", #default=True,
                        help='apply SQ on input data')
    parser.add_argument('--wt_qnt', action="store_true", #default=True,
                        help='apply SQ on weight')
    parser.add_argument('--bits', type=tuple, default=(8,8),
                        help='quantization bits of each layer')
    parser.add_argument('--wf', action="store_true", #default=True, 
                        help='write result to file')
    parser.add_argument('--pretrained', action="store_true", #default=True,
                        help='use pretrained model')
    parser.add_argument('--inf_time', action="store_true", #default=True,
                        help='record inference time')
    parser.add_argument('--print_result', action="store_true", #default=True,
                        help='')
    parser.add_argument('--fast', action="store_true", #default=True,
                        help='no need to download datasets, use data already quantized by EPQ.')
    parser.add_argument('--f', type=str, default='result.txt', help='path of result file')
    parser.add_argument('--K', type=int, default=2, help='number of hops')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_result', action="store_true", #default=True,
                        help='save result to file')
    parser.add_argument('--pre', type=str, default=None, help='type of preprocessing')
    parser.add_argument('--quant_type', type=str, default=None, help='type of quantization')
    
    args = parser.parse_args()
    print(args)
    return args

def dataProcess(dataset):
    transform = [T.SIGN(args.K)] if args.pre is not None else []
    if dataset == 'Reddit':
        from torch_geometric.datasets import Reddit
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../', 'data', 'Reddit')
        dataset = Reddit(path)
    elif dataset == 'Amazon2M':
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name = "ogbn-products", root = '../data/') 
    elif dataset == 'Actor':
        from torch_geometric.datasets import Actor
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../', 'data', 'Actor')
        transform = [T.NormalizeFeatures()] + transform
        dataset = Actor(path, transform=T.Compose(transform))
    elif dataset in ['Cornell', 'Texas', 'Wisconsin']:
        from torch_geometric.datasets import WebKB
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../', 'data', dataset)
        transform = [T.NormalizeFeatures()] + transform
        dataset = WebKB(path, dataset, transform=T.Compose(transform))
    elif dataset in ['Squirrel', 'Chameleon']:
        from torch_geometric.datasets import WikipediaNetwork
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../', 'data', dataset)
        transform = [T.NormalizeFeatures()] + transform
        dataset = WikipediaNetwork(path, dataset, geom_gcn_preprocess=True, transform=T.Compose(transform))
    else:
        from torch_geometric.datasets import Planetoid
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../', 'data', dataset)
        transform = T.Compose([T.NormalizeFeatures()] + transform)
        dataset = Planetoid(path, dataset, transform=transform)
    data = dataset[0]

    if args.quant_type is None:
        if args.pre == 'SGC':
            data.x = data[f'x{args.K}']
            for i in range(1, args.K+1):
                del data[f'x{i}']
        elif args.pre == 'SIGN':
            x_list = [data.x]
            for i in range(1, args.K+1):
                x_list.append(data[f'x{i}'])
            data.x = torch.cat(x_list, dim=-1)
            for i in range(1, args.K+1):
                del data[f'x{i}']

    return data, dataset

def train(data,model,optimizer):
    model.train()
    optimizer.zero_grad()
    loss = F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask])
    loss_value = loss.item()
    loss.backward()
    optimizer.step()

    return loss_value

@torch.no_grad()
def test(data, model):

    model.eval()
    logits, accs, specs, senses, precs, f1s = model(data), [], [], [], [], []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        # Compute confusion matrix
        y_true_np = data.y[mask].cpu()
        y_pred_np = pred.cpu()
        # Compute Accuracy
        accuracy = accuracy_score(y_true_np, y_pred_np)

        # Compute Precision, Recall, F1 Score for each class and micro average
        precision = precision_score(y_true_np, y_pred_np, average='micro', zero_division=0)
        recall = recall_score(y_true_np, y_pred_np, average='micro', zero_division=0)
        micro_f1 = f1_score(y_true_np, y_pred_np, average='micro')

        # Compute Confusion Matrix
        conf_matrix = confusion_matrix(y_true_np, y_pred_np)

        # Compute Specificity for each class
        specificity = []
        for i in range(conf_matrix.shape[0]):
            tn = np.sum(np.delete(np.delete(conf_matrix, i, axis=0), i, axis=1))
            fp = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
            specificity.append(tn / (tn + fp))

        specificity = np.mean(specificity)

        accs.append(accuracy)
        specs.append(specificity)
        senses.append(recall)
        precs.append(precision)
        f1s.append(micro_f1)
        
    return accs, specs, senses, precs, f1s

if __name__ == "__main__":
    # torch.cuda.memory._record_memory_history(max_entries=100000)
    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)  #DEBUG < INFO < WARNING < ERROR < CRITICAL
    args = argParse()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    block_size = args.block_size
    n_cents = args.ncents
    data, dataset = dataProcess(args.dataset)
    num_classes = data.num_classes if args.fast else dataset.num_classes
    data_size = data.size if args.fast else data.x.size()
    if args.dataset in ["Actor", "Cornell", "Texas", "Wisconsin", "Squirrel", "Chameleon"]:
        mask_idx = args.seed % data.train_mask.size(1)
        data.train_mask = data.train_mask[:, mask_idx].nonzero(as_tuple=False).view(-1)
        data.val_mask = data.val_mask[:, mask_idx].nonzero(as_tuple=False).view(-1)
        data.test_mask = data.test_mask[:, mask_idx].nonzero(as_tuple=False).view(-1)
    data = data.to(device)


    if args.pre is not None:
        input_size = data_size[1] * (args.K+1) if args.pre == 'SIGN' and args.quant_type is not None else data_size[1]
        input_size = data_size[1] * (args.K*2+1) if args.pre == 'H2GC' and args.quant_type is not None else input_size
        model = MLP(input_size, args.hidden, num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        if args.model == 'GCN':
            model = GCN(data_size[1], args.hidden, num_classes).to(device)
            optimizer = torch.optim.Adam([
                dict(params=model.conv1.parameters(), weight_decay=args.wd),
                dict(params=model.conv2.parameters(), weight_decay=0)
            ], lr=args.lr)  # Only perform weight-decay on first convolution.
        else:
            logger.error(f'{args.model} is not support yet.')
            sys.exit(0)

    
    save = bool(0)
    load = not save
    wt_qnt = args.wt_qnt
    act_qnt = args.act_qnt
    pqnt = args.fast or args.pqnt
    
    layer0_input_qparam = QParam(
            pqnt=pqnt, 
            n_centroids=args.ncents, 
            block_size=args.block_size,
            batch_size=args.batch_size,
            mini_batch=args.mini_batch,
            save=save,
            load=load,
            path=args.path,
    )
    
    layer0_act_qparam = QParam(
            sqnt=act_qnt,
            bits=args.bits,
            p=0.3, 
    )
    
    layer0_wt_qparam = QParam(
            sqnt=wt_qnt,
            bits=args.bits,
            p=0.3,
    )
    
    layer0_qparam = Layer_Qparam(input_qparam=layer0_input_qparam, wt_qparam=layer0_wt_qparam, act_qparam=layer0_act_qparam)
    
    layer1_input_qparam = QParam()
    
    layer1_act_qparam = QParam(
            sqnt=act_qnt,
            bits=args.bits,
            p=0.3, 
    )
    
    layer1_wt_qparam = QParam(
            sqnt=wt_qnt,
            bits=args.bits,
            p=0.3,
    )
    
    layer1_qparam = Layer_Qparam(input_qparam=layer1_input_qparam, wt_qparam=layer1_wt_qparam, act_qparam=layer1_act_qparam)
    
    qnt_param = {'layer_0':layer0_qparam, 'layer_1':layer1_qparam}
    quant_framework(model, **qnt_param)
    model = model.to(device)
    
    if pqnt:
        pact = ActPQ(
            model, 
            module_name="GCN", 
            n_centroids=layer0_input_qparam.n_centroids, 
            block_size=layer0_input_qparam.block_size, 
            try_cluster=layer0_input_qparam.try_cluster, 
            n_iter=layer0_input_qparam.n_iter,
            eps=layer0_input_qparam.eps,
            load=True,
            path=layer0_input_qparam.path,
            mini_batch=layer0_input_qparam.mini_batch,
            batch_size=layer0_input_qparam.batch_size,
            pre=args.pre,
        )
        data.x = pact.input_quant(data.x).to(device)
        del pact

        if args.pre and args.quant_type == 'EPQ':
            data = T.SIGN(args.K)(data)
            if args.pre == 'SIGN':
                x_list = [data.x]
                for i in range(1, args.K+1):
                    x_list.append(data[f'x{i}'])
                data.x = torch.cat(x_list, dim=-1).clone().to(device)
                for i in range(1, args.K):
                    del data[f'x{i}']
            elif args.pre == 'SGC':
                data.x = data[f'x{args.K}'].clone().to(device)
                for i in range(1, args.K+1):
                    del data[f'x{i}']
            elif args.pre == 'H2GC':
                data = data.to('cpu')
                data = h2gc_sparse(data, K=args.K, num_clusters=dataset.num_classes)
                data = data.to(device)
                x_list = [data.x]
                for i in range(1, args.K*2+1):
                    x_list.append(data[f'x{i}'])
                data.x = torch.cat(x_list, dim=-1).clone().to(device)
                for i in range(1, args.K*2+1):
                    del data[f'x{i}']

    result_dict = dict()
    result_dict["max_val_acc"] = 0
    result_dict["max_val_sens"] = 0
    result_dict["max_val_spec"] = 0
    result_dict["max_val_prec"] = 0
    result_dict["max_val_f1"] = 0
    result_dict["max_test_acc"] = 0
    result_dict["max_test_sens"] = 0
    result_dict["max_test_spec"] = 0
    result_dict["max_test_prec"] = 0
    result_dict["max_test_f1"] = 0
    result_dict["best_epoch"] = 0
    result_dict["training_loss"] = []
    result_dict["validation_loss"] = []
    result_dict["memory"] = []
    result_dict["time"] = []

    best_val_acc = 0
    for epoch in range(1, args.epoch+1):
        torch.cuda.reset_peak_memory_stats(device)
        start_time = time.time()
        training_loss = train(data, model, optimizer)
        accs, specs, senses, precs, f1s = test(data, model)

        if accs[1] > best_val_acc:
            best_val_acc = accs[1]
            result_dict["best_epoch"] = epoch
            result_dict["max_val_acc"] = best_val_acc
        result_dict["max_val_sens"] = max(senses[1], result_dict["max_val_sens"])
        result_dict["max_val_spec"] = max(specs[1], result_dict["max_val_spec"])
        result_dict["max_val_prec"] = max(precs[1], result_dict["max_val_prec"])
        result_dict["max_val_f1"] = max(f1s[1], result_dict["max_val_f1"])
        result_dict["max_test_acc"] = max(accs[2], result_dict["max_test_acc"])
        result_dict["max_test_sens"] = max(senses[2], result_dict["max_test_sens"])
        result_dict["max_test_spec"] = max(specs[2], result_dict["max_test_spec"])
        result_dict["max_test_prec"] = max(precs[2], result_dict["max_test_prec"])
        result_dict["max_test_f1"] = max(f1s[2], result_dict["max_test_f1"])

        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        
        memory_usage = torch.cuda.memory_allocated(device)*2**(-20)
        end_time = time.time()
        time_taken = end_time - start_time
        result_dict["memory"].append(memory_usage)
        result_dict["time"].append(time_taken)
        if epoch % 20 == 0:
            print(f"Peak GPU Memory Usage: {memory_usage} MB")
            print(f'Time for {epoch} epoch is {time_taken}\n')
            print(log.format(epoch, accs[0], best_val_acc, result_dict["max_test_acc"]))
        result_dict["training_loss"].append(training_loss)
        val_loss = F.nll_loss(model(data)[data.val_mask], data.y[data.val_mask])
        result_dict["validation_loss"].append(val_loss.item())

    result_dict["total_time"] = sum(result_dict["time"])

    if args.save_result:
        result_dict['datetime_generated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # today_date = datetime.now().strftime('%Y%m%d')
        today_date = "20240617"
        results_dir = f"results_{today_date}"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        quantized = "EPQ" if args.quant_type == "EPQ" else "Naive"
        quantized = "PAPQ" if args.quant_type == "PAPQ" else quantized
        training_file_name = f"{results_dir}/{args.dataset}_{args.model}_{quantized}_{args.pre}_training_loss.csv"
        validation_file_name = f"{results_dir}/{args.dataset}_{args.model}_{quantized}_{args.pre}_validation_loss.csv"
        memory_file_name = f"{results_dir}/{args.dataset}_{args.model}_{quantized}_{args.pre}_cuda_memory.csv"
        time_file_name = f"{results_dir}/{args.dataset}_{args.model}_{quantized}_{args.pre}_time.csv"
        results_file_name = f"{results_dir}/{args.dataset}_{args.model}_{quantized}_{args.pre}_results.csv"

        if os.path.exists(training_file_name):
            existing_training_data = np.loadtxt(training_file_name, delimiter=',')
            existing_validation_data = np.loadtxt(validation_file_name, delimiter=',')
            existing_memory_data = np.loadtxt(memory_file_name, delimiter=',')
            existing_time_data = np.loadtxt(time_file_name, delimiter=',')
            existing_results_data = pd.read_csv(results_file_name)
            updated_training_data = np.vstack([existing_training_data, result_dict["training_loss"]])
            updated_validation_data = np.vstack([existing_validation_data, result_dict["validation_loss"]])
            updated_memory_data = np.vstack([existing_memory_data, result_dict["memory"]])
            updated_time_data = np.vstack([existing_time_data, result_dict["time"]])
            result_dict.pop("validation_loss")
            result_dict.pop("training_loss")
            result_dict.pop("memory")
            result_dict.pop("time")
            updated_results_data = pd.concat([existing_results_data, pd.DataFrame(pd.Series(result_dict)).transpose()])
        else:
            updated_training_data = np.array(result_dict["training_loss"])
            updated_validation_data = np.array(result_dict["validation_loss"])
            updated_memory_data = np.array(result_dict["memory"])
            updated_time_data = np.array(result_dict["time"])
            result_dict.pop("validation_loss")
            result_dict.pop("training_loss")
            result_dict.pop("memory")
            result_dict.pop("time")
            updated_results_data = pd.DataFrame(pd.Series(result_dict)).transpose()

        np.savetxt(training_file_name, updated_training_data, delimiter=',')
        np.savetxt(validation_file_name, updated_validation_data, delimiter=',')
        np.savetxt(memory_file_name, updated_memory_data, delimiter=',')
        np.savetxt(time_file_name, updated_time_data, delimiter=',')
        updated_results_data.to_csv(results_file_name, index=False)
