import os.path as osp
import os
import torch
import argparse
from torch.utils.data import DataLoader
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.seed import seed_everything
from h2gc import h2gc
from datetime import datetime
import pandas as pd
import numpy as np
import logging
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logger = logging.getLogger("")
logger.setLevel(logging.INFO)  #DEBUG < INFO < WARNING < ERROR < CRITICAL


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
    parser.add_argument('--dataset', type=str, default='Reddit')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--mini_batch', action="store_true", default=True, 
                        help='apply batch method')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='number of nodes in each batch')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--save_result', action="store_true", #default=True,
                        help='save result to file')
    parser.add_argument('--pre', type=str, default='SIGN')
    parser.add_argument('--K', type=int, default=2)
    parser.add_argument('--n_clusters', type=int, default=None)
    parser.add_argument('--n_hidden', type=int, default=1024)
#    parser.add_argument('--verbose', type=str, default='INFO')
    
    args = parser.parse_args()
    print(args)
    return args


def dataProcess(dataset):
    transform = [T.SIGN(args.K)] if args.pre == 'SIGN' else []
    if dataset == 'Reddit':
        from torch_geometric.datasets import Reddit
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../', 'data', 'Reddit')
        dataset = Reddit(path, transform=T.Compose(transform))
    elif dataset == 'Flickr':
        from torch_geometric.datasets import Flickr
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../', 'data', 'Flickr')
        transform = [T.NormalizeFeatures()] + transform
        dataset = Flickr(path, transform=T.Compose(transform))
    elif dataset == 'Yelp':
        from torch_geometric.datasets import Yelp
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../', 'data', 'Yelp')
        dataset = Yelp(path, transform=T.Compose(transform))
    else:
        from torch_geometric.datasets import Planetoid
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../', 'data', dataset)
        dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())

    data = dataset[0]

    if args.pre == 'SIGN':
        x_list = [data.x]
        for i in range(1, args.K+1):
            x_list.append(data[f'x{i}'])
        data.x = torch.cat(x_list, dim=-1)
        for i in range(1, args.K+1):
            del data[f'x{i}']
    
    return data, dataset

args = argParse()
seed_everything(args.seed)
device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
K = args.K
data, dataset = dataProcess(args.dataset)
num_node_features = dataset.num_node_features
num_classes = dataset.num_classes
num_clusters = num_classes if args.n_clusters is None else args.n_clusters

if args.pre == "H2GC":
    data = h2gc(data, K=K, num_clusters=num_clusters)
    data = data.to('cpu')
    x_list = [data.x]
    for i in range(1, K*2+1):
        x_list.append(data[f'x{i}'])
    data.x = torch.cat(x_list, dim=-1)
    for i in range(1, K*2+1):
        del data[f'x{i}']

data = data.to(device)

train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
val_idx = data.val_mask.nonzero(as_tuple=False).view(-1)
test_idx = data.test_mask.nonzero(as_tuple=False).view(-1)

train_loader = DataLoader(train_idx, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_idx, batch_size=32 * args.batch_size)
test_loader = DataLoader(test_idx, batch_size=32 * args.batch_size)

classification_type = "multiclass" if args.dataset in ["Reddit", "Flickr"] else "multilabel"
criterion = torch.nn.BCELoss() if classification_type == 'multilabel' else torch.nn.NLLLoss()

data_size = data.x.size()
input_size = data_size[1]
model = MLP(input_size, args.n_hidden, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)


def train(data, loader):
    model.train()

    total_loss = total_examples = 0
    for idx in loader:
        x = data.x[idx].to(device)
        y = data.y[idx].to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * idx.numel()
        total_examples += idx.numel()

    return total_loss / total_examples


@torch.no_grad()
def test(data, loader):
    model.eval()
    outs = []
    ys = []
    for idx in loader:
        xs = data.x[idx].to(device)
        y = data.y[idx].to(device)

        out = model(xs)
        
        outs.append(out)
        ys.append(y)
    
    out = torch.cat(outs, dim=0)
    y = torch.cat(ys, dim=0)

    # Compute confusion matrix
    y_true_np = y.unsqueeze(-1).cpu().numpy()
    y_pred_np = out.argmax(dim=-1, keepdim=True).cpu().numpy()

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

    return accuracy, specificity, recall, precision, micro_f1


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
result_dict["memory"] = []
result_dict["time"] = []

best_val_acc = test_acc = best_test_acc = 0
for epoch in range(1, args.epoch+1):
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()
    training_loss = train(data, train_loader)
    train_acc, train_spec, train_sense, train_prec, train_f1 = test(data, train_loader)
    val_acc, val_spec, val_sense, val_prec, val_f1 = test(data, val_loader)
    test_acc, test_spec, test_sense, test_prec, test_f1 = test(data, test_loader)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = test_acc
        result_dict["best_epoch"] = epoch
        result_dict["max_val_acc"] = best_val_acc
    result_dict["max_val_sens"] = max(val_sense, result_dict["max_val_sens"])
    result_dict["max_val_spec"] = max(val_spec, result_dict["max_val_spec"])
    result_dict["max_val_prec"] = max(val_prec, result_dict["max_val_prec"])
    result_dict["max_val_f1"] = max(val_f1, result_dict["max_val_f1"])
    result_dict["max_test_acc"] = max(test_acc, result_dict["max_test_acc"])
    result_dict["max_test_sens"] = max(test_sense, result_dict["max_test_sens"])
    result_dict["max_test_spec"] = max(test_spec, result_dict["max_test_spec"])
    result_dict["max_test_prec"] = max(test_prec, result_dict["max_test_prec"])
    result_dict["max_test_f1"] = max(test_f1, result_dict["max_test_f1"])
    memory_usage = torch.cuda.max_memory_allocated(device)*2**(-20)
    end_time = time.time()
    time_taken = end_time - start_time
    if epoch % 20 == 0:
        print(f'F1 Score: {train_f1:.4f}, {val_f1:.4f}, {test_f1:.4f}')
        print(f'Best F1 Score: {result_dict["max_val_f1"]:.4f}, {result_dict["max_test_f1"]:.4f}')
        print(f'Accuracy: {train_acc:.4f}, {val_acc:.4f}, {test_acc:.4f}')
        print(f'Best Accuracy: {result_dict["max_val_acc"]:.4f}, {result_dict["max_test_acc"]:.4f}')
        print(f"Peak GPU Memory Usage: {memory_usage} MB")
        print(f"Time: {time_taken}")
        print(f'Epoch {epoch:02d}, Training Loss: {training_loss:.4f}')
    result_dict["training_loss"].append(training_loss)
    # print(f'Epoch {epoch:02d}, Training Loss: {training_loss:.4f}, Validation Loss: {val_loss:.4f}')
    result_dict["memory"].append(memory_usage)
    result_dict["time"].append(time_taken)

result_dict["total_time"] = sum(result_dict["time"])

if args.save_result:
    result_dict['datetime_generated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # today_date = datetime.now().strftime('%Y%m%d')
    today_date = "20240601"
    results_dir = f"results_dense_{today_date}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    method = args.pre
    prefix = f"{args.dataset}_{method}_{args.K}_{args.n_clusters}"
    training_file_name = f"{results_dir}/{prefix}_training_loss.csv"
    memory_file_name = f"{results_dir}/{prefix}_cuda_memory.csv"
    time_file_name = f"{results_dir}/{prefix}_time.csv"
    results_file_name = f"{results_dir}/{prefix}_results.csv"

    if os.path.exists(training_file_name):
        existing_training_data = np.loadtxt(training_file_name, delimiter=',')
        existing_memory_data = np.loadtxt(memory_file_name, delimiter=',')
        existing_time_data = np.loadtxt(time_file_name, delimiter=',')
        existing_results_data = pd.read_csv(results_file_name)
        updated_training_data = np.vstack([existing_training_data, result_dict["training_loss"]])
        updated_memory_data = np.vstack([existing_memory_data, result_dict["memory"]])
        updated_time_data = np.vstack([existing_time_data, result_dict["time"]])
        result_dict.pop("training_loss")
        result_dict.pop("memory")
        result_dict.pop("time")
        updated_results_data = pd.concat([existing_results_data, pd.DataFrame(pd.Series(result_dict)).transpose()])
    else:
        updated_training_data = np.array(result_dict["training_loss"])
        updated_memory_data = np.array(result_dict["memory"])
        updated_time_data = np.array(result_dict["time"])
        result_dict.pop("training_loss")
        result_dict.pop("memory")
        result_dict.pop("time")
        updated_results_data = pd.DataFrame(pd.Series(result_dict)).transpose()

    np.savetxt(training_file_name, updated_training_data, delimiter=',')
    np.savetxt(memory_file_name, updated_memory_data, delimiter=',')
    np.savetxt(time_file_name, updated_time_data, delimiter=',')
    updated_results_data.to_csv(results_file_name, index=False)
        

