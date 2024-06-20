import argparse
import logging
import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch_geometric.transforms as T
from torch_geometric.seed import seed_everything
from h2gc import h2gc_sparse
from mlp import MLP
from datetime import datetime
import numpy as np
import pandas as pd
import os
import os.path as osp
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class MLP(torch.nn.Module):
    def __init__(self, nfeature, nhid, nclass):
        super(MLP, self).__init__()
        self.conv1 = Linear(nfeature, nhid)
        self.conv2 = Linear(nhid, nclass)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x)
        return F.log_softmax(x, dim=1)

def argParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--model', type=str, default='MLP')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--K', type=int, default=2)
    parser.add_argument('--pre', type=str, default='SGC')
    parser.add_argument('--n_clusters', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_result', action="store_true", #default=True,
                        help='save result to file')
    
    args = parser.parse_args()
    print(args)
    return args

def dataProcess(dataset):
    transform = [T.SIGN(args.K)] if args.pre in ['SIGN', 'SGC'] else []
    if dataset == 'Actor':
        from torch_geometric.datasets import Actor
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../', 'data', 'Actor')
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
    loss = F.nll_loss(model(data.x)[data.train_mask], data.y[data.train_mask])
    loss_value = loss.item()
    loss.backward()
    optimizer.step()

    return loss_value

@torch.no_grad()
def test(data, model):

    model.eval()
    logits, accs, specs, senses, precs, f1s = model(data.x), [], [], [], [], []
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
    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)  #DEBUG < INFO < WARNING < ERROR < CRITICAL
    args = argParse()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed)
    K = args.K

    data, dataset = dataProcess(args.dataset)
    num_classes = dataset.num_classes
    num_clusters = num_classes if args.n_clusters is None else args.n_clusters

    if args.pre in ['H2GC', 'HHSGC']:
        data = h2gc_sparse(data, K=K, num_clusters=num_clusters)
        data = data.to('cpu')
        if args.pre == 'HHSGC':
            data.x = torch.cat([data[f'x{(K-1)*2+1}'], data[f'x{(K-1)*2+2}']], dim=-1)
            for i in range(1, K*2+1):
                del data[f'x{i}']
        elif args.pre == 'H2GC':
            x_list = [data.x]
            for i in range(1, K*2+1):
                x_list.append(data[f'x{i}'])
            data.x = torch.cat(x_list, dim=-1)
            for i in range(1, K*2+1):
                del data[f'x{i}']

    data = data.to(device)
    if args.dataset in ["Actor", "Cornell", "Texas", "Wisconsin", "Squirrel", "Chameleon"]:
        mask_idx = args.seed % data.train_mask.size(1)
        data.train_mask = data.train_mask[:, mask_idx].nonzero(as_tuple=False).view(-1)
        data.val_mask = data.val_mask[:, mask_idx].nonzero(as_tuple=False).view(-1)
        data.test_mask = data.test_mask[:, mask_idx].nonzero(as_tuple=False).view(-1)
    data_size = data.x.size()
    input_size = data_size[1]
    model = MLP(input_size, args.n_hidden, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    model = model.to(device)
    print(model)
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
        if epoch % 50 == 0:
            print(f"Peak GPU Memory Usage: {memory_usage} MB")
            print(f'Time for {epoch} epoch is {time_taken}\n')
            print(log.format(epoch, accs[0], best_val_acc, result_dict["max_test_acc"]))
        result_dict["training_loss"].append(training_loss)
        # val_loss = F.nll_loss(model(data)[data.val_mask], data.y[data.val_mask])
        # result_dict["validation_loss"].append(val_loss.item())

    result_dict["total_time"] = sum(result_dict["time"])

    if args.save_result:
        result_dict['datetime_generated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # today_date = datetime.now().strftime('%Y%m%d')
        today_date = "20240612"
        results_dir = f"results_{today_date}"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        training_file_name = f"{results_dir}/{args.dataset}_{args.model}_{args.pre}_{args.K}_training_loss.csv"
        # validation_file_name = f"{results_dir}/{args.dataset}_{args.model}_{args.pre}_{args.K}_validation_loss.csv"
        memory_file_name = f"{results_dir}/{args.dataset}_{args.model}_{args.pre}_{args.K}_cuda_memory.csv"
        time_file_name = f"{results_dir}/{args.dataset}_{args.model}_{args.pre}_{args.K}_time.csv"
        results_file_name = f"{results_dir}/{args.dataset}_{args.model}_{args.pre}_{args.K}_results.csv"

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
