import torch
import numpy as np
from tqdm import tqdm

from data_loaders.data_loader import load_data_sets
import torchmetrics
# from sklearn.metrics import confusion_matrix
all_class_name = [
    "",
    "RIGHT",
    "KNOB",
    "CROSS",
    "THREE",
    "V",
    "ONE",
    "FOUR",
    "GRAB",
    "DENY",
    "MENU",
    "CIRCLE",
    "TAP",
    "PINCH",
    "LEFT",
    "TWO",
    "OK",
    "EXPAND",
]


batch_size = 32
workers = 4
lr = 1e-3
num_classes = 18
window_size=20
input_shape = (window_size,20,3)
device = torch.device('cuda')
def compute_energy(x):
    N, T, V, C = x.shape

    x_values= x[:,:,:,0]
    y_values = x[:, :, :, 1]
    z_values = x[:, :, :, 2]
    w=None
    for v in range(V):
        w_v=None
        for t in range(1,T):
            if w_v == None :
                w_v = torch.sqrt(( x_values[:,t,v]/x_values[:,t-1,v] -1)**2 + ( y_values[:,t,v]/y_values[:,t-1,v] -1)**2 + ( z_values[:,t,v]/z_values[:,t-1,v] -1)**2)
            else :
                w_v  += torch.sqrt((x_values[:, t, v] / x_values[:, t - 1, v] - 1) ** 2 + (
                            y_values[:, t, v] / y_values[:, t - 1, v] - 1) ** 2 + (
                                           z_values[:, t, v] / z_values[:, t - 1, v] - 1) ** 2)
        if w==None :
            w=w_v
        else :
            w+=w_v
    return w
def init_data_loader():
    train_loader, val_loader, test_loader, graph = load_data_sets(
        dataset_name="ODHG",
        window_size=window_size,
            batch_size=batch_size,
            workers=workers,
            is_segmented=True,
            binary_classes=False,
            use_data_aug=True,
            use_aug_by_sw=True
            )

    return train_loader, test_loader, val_loader, graph


def init_model(graph,num_classes,seq_len,dropout_rate=.1):
    model = STrGCN(graph, d_model=128,n_heads=8,num_classes=num_classes, dropout=dropout_rate)
    model = torch.nn.DataParallel(model).cuda()

    return model





def get_acc(score, labels):
    score = score.cpu().data.numpy()
    labels = labels.cpu().data.numpy()
    outputs = np.argmax(score, axis=1)
    return np.sum(outputs == labels) / float(labels.size)

def get_fp_rate(score,labels):
    confusion_matrix=torchmetrics.ConfusionMatrix(num_classes=num_classes)


    cnf_matrix = confusion_matrix(score.detach().cpu(), labels.detach().cpu())
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.type(torch.float)
    TN = TN.type(torch.float)

    # # Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP/(TP+FN)
    # # Specificity or true negative rate
    # TNR = TN/(TN+FP)
    # # Precision or positive predictive value
    # PPV = TP/(TP+FP)
    # # Negative predictive value
    # NPV = TN/(TN+FN)
    # # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # # False negative rate
    # FNR = FN/(TP+FN)
    # # False discovery rate
    # FDR = FP/(TP+FP)
    # # Overall accuracy
    # ACC = (TP+TN)/(TP+FP+FN+TN)
    return torch.sum(torch.nan_to_num(FPR),dim=-1)

def get_window_label(label):

    N,W=label.shape

    sum=torch.zeros((N,num_classes))
    for i in range(N):
        for t in range(W):
            sum[i, label[i, t]] += 1
    return  sum.argmax(dim=-1).cuda()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    # fold for saving trained model...
    # change this path to the fold where you want to save your pre-trained model

    train_loader, test_loader, val_loader, adjacency_matrix = init_data_loader()



    for i, batch in tqdm(enumerate(train_loader), leave=False):
        print("batch=",i)
        print(batch)

