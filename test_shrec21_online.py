import torch
import numpy as np
from tqdm import tqdm
import time
import os
from model import CoSTrGCN
from data_loaders.data_loader import load_data_sets
import torchmetrics
import json

labels = [
    "NO GESTURE",
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
lr = 1e-4
num_classes = 18
window_size=30
input_shape = (window_size,20,3)
device = torch.device('cuda')
d_model=128
n_heads=8
lr = 1e-3
betas=(.9,.98)
epsilon=1e-9
weight_decay=5e-4
optimizer_params=(lr,betas,epsilon,weight_decay)
Max_Epochs = 500
Early_Stopping = 25
dropout_rate=.3
num_classes=2
stride=window_size-20
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
    window_size=window_size,
        batch_size=batch_size,
        workers=workers,
        is_segmented=False,
        binary_classes=True,
        use_data_aug=False,
        use_aug_by_sw=False
        )

    return train_loader, val_loader, test_loader, graph


def init_model(graph, optimizer_params, labels,num_classes,dropout_rate=.1):
    model = CoSTrGCN(graph, optimizer_params, labels, d_model=128,n_heads=8,num_classes=num_classes, dropout=dropout_rate)
    

    return model





def get_acc(score, labels):
    score = score.cpu().data.numpy()
    labels = labels.cpu().data.numpy()
    outputs = np.argmax(score, axis=1)
    return np.sum(outputs == labels) / float(labels.size)

def get_fp_rate(score,labels):
    confusion_matrix=torchmetrics.StatScores(num_classes=num_classes,reduce="micro")


    TP, FP, TN, FN, SUP = confusion_matrix(score, labels)
    # FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    # FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    # TP = np.diag(cnf_matrix)
    # TN = cnf_matrix.sum() - (FP + FN + TP)

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

    sum=torch.zeros((1,num_classes))
    for t in range(N):
        sum[0,label[t]] += 1
    out=sum.argmax(dim=-1)
    return  out 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    # fold for saving trained model...
    # change this path to the fold where you want to save your pre-trained model
    model_fold = "./models/costr_gcn/online_model_checkpoints"
    try:
        os.mkdir(model_fold)
    except:
        pass

    train_loader, val_loader, test_loader, graph = init_data_loader()



    # .........inital model
    print("\n loading model.............")
    model = model = CoSTrGCN.load_from_checkpoint(checkpoint_path="./models/CoSTrGCN-SHREC21_2022-09-04_22_27_16/best_model-128-8-v1.ckpt",adjacency_matrix=graph, optimizer_params=optimizer_params, labels=labels, d_model=128,n_heads=8,num_classes=num_classes, dropout=dropout_rate)
    # model_solver = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # # ........set loss
    criterion = torch.nn.CrossEntropyLoss()



    # parameters recording training log


    f1_score=torchmetrics.F1Score(num_classes=num_classes)

    jaccard = torchmetrics.JaccardIndex(num_classes=num_classes)
    avg_precision = torchmetrics.AveragePrecision(num_classes=num_classes)
    eps=1e-1
    start_time = time.time()
    #         # ***********evaluation***********
    print("*"*10,"Testing","*"*10)
    results=[]
    with torch.no_grad():
        val_loss = 0
        val_f1 = 0
        val_jaccard=0
        val_fp_rate=0
        val_avg_precision=0
        score_list = None
        label_list = None
        acc_sum = 0
        # model.eval()
        val_loss_epoch = 0
        val_jaccard_epoch=0
        val_fp_rate_epoch=0
        val_avg_precision_epoch=0
        val_f1_epoch = 0
        for i, batch in tqdm(enumerate(test_loader), leave=False):
            # print("batch=",i)
            x,y,index=batch
            
            y=torch.stack(y)
            N, T, V, C = x.shape




            
            score_list = None
            label_list = None   
            num_windows=T-window_size // window_size

            for t in tqdm(range(0,T-window_size+1,stride), leave=False):
                # print(i)
                window=x[:,t:t+window_size]
                
                label_l=y[t:t+window_size]
                # print(label_l)
                label=get_window_label(label_l)

                # window = x[:,t: t+window_size].clone()
                
                # if t < 2*stride :
                #     continue
                # window_i_m_2 = x[:,(t-2*stride): (t-2*stride)+window_size].clone()
                # window_i_m_1 = x[:,(t-1*stride):(t-1*stride)+window_size ].clone()
                # window_i = x[:,t: t+window_size].clone()
                # window_i_p_1 = x[:,t+1*stride: t+1*stride+window_size].clone()
                # window_i_p_2 = x[:,t+2*stride: (t+2*stride)+window_size].clone()

                # w_1=compute_energy(window_i_m_2)
                
                # w_2=compute_energy(window_i_m_1)
                # w_3=compute_energy(window_i)
                # w_4=compute_energy(window_i_p_1)
                # w_5=compute_energy(window_i_p_2)
                # d_wi=(w_4-w_2)/((t+1*stride)-(t-1*stride))
                # d_wi_m_1=(w_3-w_1)/(t-(t-2*stride))
                # d_wi_p_1=(w_5-w_3)/((t+2*stride)-t)
                # if d_wi < eps and d_wi_m_1 > 0 and d_wi_p_1 < 0 :

                score = model(window)

                score_list_labels= torch.argmax(torch.nn.functional.softmax(score, dim=-1), dim=-1)
                results.append({
                    "prediction":score_list_labels.item(),
                    "label":label.item(),
                })

                if score_list is None:
                    score_list = score
                    label_list = label
                else:
                    score_list = torch.cat((score_list, score), 0)
                    label_list = torch.cat((label_list, label), 0)


            loss = criterion(score_list.detach().cpu(), label_list.detach().cpu())
            score_list_labels= torch.argmax(torch.nn.functional.softmax(score_list, dim=-1), dim=-1)
            val_f1_step= f1_score(score_list_labels.detach().cpu(), label_list.detach().cpu())
            val_jaccard_step= jaccard(score_list_labels.detach().cpu(), label_list.detach().cpu())
            val_fp_rate_step= get_fp_rate(score_list_labels.detach().cpu(), label_list.detach().cpu())
            val_avg_precision_step=avg_precision(score_list.detach().cpu(), label_list.detach().cpu())
            val_f1_epoch += val_f1_step
            val_jaccard_epoch += val_jaccard_step
            val_fp_rate_epoch += val_fp_rate_step
            val_avg_precision_epoch+=val_avg_precision_step
            val_loss += loss
            print("*** SHREC  21"
                "val_loss_step: %.6f,"
                "val_F1_step: %.6f ***,"
                "val_jaccard_step: %.6f ***"
                "val_fp_rate_step: %.6f ***"
                "val_avg_precision_step: %.6f ***"
                % ( loss, val_f1_step,val_jaccard_step, val_fp_rate_step,val_avg_precision_step))
        with open('results_analysis.json',mode="w") as f:
            json.dump(results,f,indent=2)
        val_loss = val_loss / (float(i + 1))
        val_f1 = val_f1_epoch.item() / (float(i + 1))
        val_jaccard = val_jaccard_epoch / (float(i + 1))
        val_fp_rate = val_fp_rate_epoch / (float(i + 1))
        val_avg_precision = val_avg_precision_epoch / (float(i + 1))
        print("*** SHREC 21, "
                "val_loss: %.6f,"
                "val_F1: %.6f ***,"
                "val_jaccard: %.6f ***"
                "val_fp_rate: %.6f ***"
                "val_avg_precision_rate: %.6f ***"
                % (val_loss, val_f1,val_jaccard, val_fp_rate, val_avg_precision))


