#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os.path as osp
import os
import torch
import argparse
from glora_benchmark import SAGE
from torch_geometric.loader import NeighborSampler
from datetime import datetime
import pandas as pd
import numpy as np
import logging
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logger = logging.getLogger("")
logger.setLevel(logging.INFO)  #DEBUG < INFO < WARNING < ERROR < CRITICAL

# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6'

def argParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Reddit')
    parser.add_argument('--model', type=str, default='GS-mean')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--n_hidden', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--glora_rank', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='number of nodes in each batch')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_result', action="store_true", #default=True,
                        help='save result to file')
    
    args = parser.parse_args()
    print(args)
    return args


def dataProcess(dataset, use_gdc=False):
    if dataset == 'Reddit':
        from torch_geometric.datasets import Reddit
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../', 'data', 'Reddit')
        dataset = Reddit(path)
    elif dataset == 'Flickr':
        from torch_geometric.datasets import Flickr
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../', 'data', 'Flickr')
        dataset = Flickr(path)
    elif dataset == 'Amazon2M':
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name = "ogbn-products", root = '../data/') 
    else:
        pass
    
    return dataset

def get_sampler(dataset_name, data, batch_size):
    sample_sizes = {
        'Reddit': ([25, 10], [25]),
        'Flickr': ([50, 25], [50]),
        }
    train_loader_sizes, subgraph_loader_sizes = sample_sizes[dataset_name]
    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                    sizes=train_loader_sizes, 
                                    batch_size=batch_size, shuffle=True,
                                    num_workers=12)
    subgraph_loader = NeighborSampler(data.edge_index, node_idx=None,
                                        sizes=subgraph_loader_sizes,
                                        batch_size=batch_size, shuffle=False,
                                        num_workers=12)
    return train_loader, subgraph_loader


args = argParse()
torch.manual_seed(args.seed)
dataset = dataProcess(args.dataset)
data = dataset[0]

train_loader, subgraph_loader = get_sampler(args.dataset, data, args.batch_size)

device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
model = SAGE(data.x.size()[1], args.n_hidden, dataset.num_classes, args.glora_rank)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

x = data.x.to(device)
y = data.y.squeeze().to(device)

criterion = torch.nn.NLLLoss()

def train(x):
    model.train()

    total_loss = 0
    total_memory = 0
    count = 0
    for batch_size, n_id, edge_index in train_loader:
        count += 1
        torch.cuda.reset_peak_memory_stats(device)
        adjs = [adj.to(device) for adj in edge_index]
        optimizer.zero_grad()
        out = model([x[n_id], adjs])
        loss = criterion(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_memory += torch.cuda.max_memory_allocated(device)*2**(-20)
    
    memory_usage = total_memory / count
    loss = total_loss / len(train_loader)

    return loss, memory_usage


@torch.no_grad()
def test(x):
    model.eval()
    out = model.inference(
        x,
        edge_index=data.edge_index,
        subgraph_loader=subgraph_loader, 
        device=device
    )

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)
    accs, specs, senses, precs, f1s = [], [], [], [], []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        pred = y_pred[mask]
        # Compute confusion matrix
        y_true_np = y_true[mask].cpu()
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
# result_dict["validation_loss"] = []
result_dict["memory"] = []
result_dict["time"] = []

best_val_acc = 0
for epoch in range(1, args.epoch+1):
    start_time = time.time()
    training_loss, memory_usage = train(x)
    print(f'Epoch {epoch:02d}, Loss: {training_loss:.4f}')
    accs, specs, senses, precs, f1s = test(x)
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

    print(f'F1 Score: {f1s[0]:.4f}, {f1s[1]:.4f}, {f1s[2]:.4f}')
    print(f'Train: {accs[0]:.4f}, Val: {accs[1]:.4f}, '
            f'Test: {accs[2]:.4f}')
    print(f"Peak GPU Memory Usage: {memory_usage} MB")
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time: {time_taken}")
    result_dict["training_loss"].append(training_loss)
    # val_loss = train(x, is_validation=True)
    # result_dict["validation_loss"].append(val_loss)
    result_dict["memory"].append(memory_usage)
    result_dict["time"].append(time_taken)

result_dict["total_time"] = sum(result_dict["time"])

if args.save_result:
    result_dict['datetime_generated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # today_date = datetime.now().strftime('%Y%m%d')
    today_date = "20240608"
    results_dir = f"results_dense_{today_date}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    training_file_name = f"{results_dir}/{args.dataset}_{args.model}_{args.glora_rank}_training_loss.csv"
    # validation_file_name = f"{results_dir}/{args.dataset}_{args.model}_{args.glora_rank}_validation_loss.csv"
    memory_file_name = f"{results_dir}/{args.dataset}_{args.model}_{args.glora_rank}_cuda_memory.csv"
    time_file_name = f"{results_dir}/{args.dataset}_{args.model}_{args.glora_rank}_time.csv"
    results_file_name = f"{results_dir}/{args.dataset}_{args.model}_{args.glora_rank}_results.csv"

    if os.path.exists(training_file_name):
        existing_training_data = np.loadtxt(training_file_name, delimiter=',')
        # existing_validation_data = np.loadtxt(validation_file_name, delimiter=',')
        existing_memory_data = np.loadtxt(memory_file_name, delimiter=',')
        existing_time_data = np.loadtxt(time_file_name, delimiter=',')
        existing_results_data = pd.read_csv(results_file_name)
        updated_training_data = np.vstack([existing_training_data, result_dict["training_loss"]])
        # updated_validation_data = np.vstack([existing_validation_data, result_dict["validation_loss"]])
        updated_memory_data = np.vstack([existing_memory_data, result_dict["memory"]])
        updated_time_data = np.vstack([existing_time_data, result_dict["time"]])
        # result_dict.pop("validation_loss")
        result_dict.pop("training_loss")
        result_dict.pop("memory")
        result_dict.pop("time")
        updated_results_data = pd.concat([existing_results_data, pd.DataFrame(pd.Series(result_dict)).transpose()])
    else:
        updated_training_data = np.array(result_dict["training_loss"])
        # updated_validation_data = np.array(result_dict["validation_loss"])
        updated_memory_data = np.array(result_dict["memory"])
        updated_time_data = np.array(result_dict["time"])
        # result_dict.pop("validation_loss")
        result_dict.pop("training_loss")
        result_dict.pop("memory")
        result_dict.pop("time")
        updated_results_data = pd.DataFrame(pd.Series(result_dict)).transpose()

    np.savetxt(training_file_name, updated_training_data, delimiter=',')
    # np.savetxt(validation_file_name, updated_validation_data, delimiter=',')
    np.savetxt(memory_file_name, updated_memory_data, delimiter=',')
    np.savetxt(time_file_name, updated_time_data, delimiter=',')
    updated_results_data.to_csv(results_file_name, index=False)
        

