import torch
import numpy as np
from tqdm import tqdm
import time
import os
from model import CoSTrGCN
from data_loaders.data_loader import load_data_sets
import torchmetrics
import json
from torchmetrics.functional import jaccard_index, confusion_matrix
import matplotlib.pyplot as plt
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
window_size=50
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
num_classes=18
stride=window_size-10
cnf_matrix=None
with open('thresholds_best.json',mode="r") as f:
    thresholds=json.load(f)

def plot_confusion_matrix(cnf_matrix, labels, filename,mode="eps",eps=1e-5) :
    import seaborn as sn
    confusion_matrix_sum_vec= torch.sum(cnf_matrix,dim=1) +eps
    
    confusion_matrix_percentage=(cnf_matrix /  confusion_matrix_sum_vec.view(-1,1) )

    plt.figure(figsize = (20,18))
    sn.heatmap(confusion_matrix_percentage.cpu().numpy(), annot=True,cmap="coolwarm", xticklabels=labels,yticklabels=labels)
    sn.set(font_scale=1.4)
    plt.savefig(filename,format=mode)
    plt.close()

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


def compute_std(x,label):
    N, T, V, C = x.shape

    s=torch.mean(x,dim=1)
    # print(label_l)
    # print(torch.std(s,dim=-1))
    # print(torch.std(s,dim=1))
    # print(torch.std(s))

def velocity(x):
    N, T, V, C = x.shape

    start_x_values= x[:,0,:,0]
    start_y_values = x[:, 0, :, 1]
    start_z_values = x[:, 0, :, 2]
    end_x_values= x[:,-1,:,0]
    end_y_values = x[:, -1, :, 1]
    end_z_values = x[:, -1, :, 2]
    distance=torch.sqrt( (end_x_values-start_x_values)**2+(end_y_values-start_y_values)**2+(end_z_values-start_z_values)**2)

    v=distance / T
    return v

def init_data_loader():
    train_loader, val_loader, test_loader, graph = load_data_sets(
    window_size=window_size,
        batch_size=batch_size,
        workers=workers,
        is_segmented=False,
        binary_classes=False,
        use_data_aug=False,
        use_aug_by_sw=False
        )

    return train_loader, val_loader, test_loader, graph







def get_acc(score, labels):
    score = score.cpu().data.numpy()
    labels = labels.cpu().data.numpy()
    outputs = np.argmax(score, axis=1)
    return np.sum(outputs == labels) / float(labels.size)

def get_fp_rate(scores,labels):
    stat_scores=torchmetrics.StatScores(num_classes=num_classes,reduce="micro")


    TP, FP, TN, FN, SUP = stat_scores(scores, labels)
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
    FPR = FP/(TP+FN)
    # # False negative rate
    # FNR = FN/(TP+FN)
    # # False discovery rate
    # FDR = FP/(TP+FP)
    # # Overall accuracy
    # ACC = (TP+TN)/(TP+FP+FN+TN)
    print(FPR)
    return torch.mean(torch.nan_to_num(FPR),dim=-1)

def get_jaccard(scores,labels):
    unique=torch.unique(torch.cat([torch.unique(scores),torch.unique(labels)]))
    
    
    return jaccard_index(scores.unsqueeze(-1),labels,average="weighted",num_classes=num_classes)

def get_detection_rate(scores,labels):
    stat_scores=torchmetrics.StatScores(num_classes=num_classes,reduce="micro")


    TP, FP, TN, FN, SUP = stat_scores(scores, labels)
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
    det_rate = TP/(TP+FN)
    # # False negative rate
    # FNR = FN/(TP+FN)
    # # False discovery rate
    # FDR = FP/(TP+FP)
    # # Overall accuracy
    # ACC = (TP+TN)/(TP+FP+FN+TN)
    return torch.mean(torch.nan_to_num(det_rate),dim=-1)
def get_window_label(label):
    N,W=label.shape
    
    sum=torch.zeros((1,num_classes))
    for t in range(N):
        sum[0,label[t]] += 1
    out=sum.argmax(dim=-1)
    
    dominant_class_ratio=sum.max(dim=-1)[0].item() / N

    return  out if dominant_class_ratio > 0.5 else torch.tensor([0])

def load_model(graph):
    
    model = CoSTrGCN.load_from_checkpoint(checkpoint_path="./models/CoSTrGCN-SHREC21_2022-09-10_14_32_38/best_model-128-8-v1.ckpt",adjacency_matrix=graph, optimizer_params=optimizer_params, labels=labels, d_model=128,n_heads=8,num_classes=num_classes, dropout=dropout_rate)
    model.eval()
    return model
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":


    train_loader, val_loader, test_loader, graph = init_data_loader()



    # .........inital model
    print("\n loading model.............")
    model=load_model(graph)
    
    # model_solver = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # # ........set loss
    criterion = torch.nn.CrossEntropyLoss()



    # parameters recording training log


    f1_score=torchmetrics.F1Score(num_classes=num_classes)

    # jaccard = torchmetrics.JaccardIndex(num_classes=num_classes)
    avg_precision = torchmetrics.AveragePrecision(num_classes=num_classes)
    eps=1e-4
    start_time = time.time()
    #         # ***********evaluation***********
    print("*"*10,"Testing","*"*10)
    results=[]
    max_l=0
    min_l=10000

    for i, batch in tqdm(enumerate(train_loader), leave=False):
        max_l=max(batch[0].shape[1],max_l)
        min_l=min(batch[0].shape[1],min_l)
    for i, batch in tqdm(enumerate(test_loader), leave=False):
        max_l=max(batch[0].shape[1],max_l)
        min_l=min(batch[0].shape[1],min_l)

    print(min_l,max_l)
    input()

    with torch.no_grad():
        
        val_loss = 0
        val_f1 = 0
        val_jaccard=0
        val_fp_rate=0
        val_avg_precision=0
        val_det_rate=0
        score_list = None
        label_list = None
        window_label_list = None
        window_score_list = None
        acc_sum = 0
        # model.eval()
        val_loss_epoch = 0
        val_jaccard_epoch=0
        val_fp_rate_epoch=0
        val_avg_precision_epoch=0
        val_f1_epoch = 0
        val_det_rate_epoch = 0
        for i, batch in tqdm(enumerate(test_loader), leave=False):
            x,y,index=batch
            
            y=torch.stack(y)
            N, T, V, C = x.shape

            
            score_list = None
            label_list = None   
            window_score_list = None
            window_label_list = None
            num_windows=T-window_size // window_size
            res=[]
            for t in tqdm(range(0,T-window_size,stride), leave=False):
                window=x[:,t:t+window_size]
                

                label_l=y[t:t+window_size]
                label=get_window_label(label_l)
                # print(label)
                # if t < 2*stride :
                #     continue
                # window_i_m_2 = x[:,t-2*stride: t-2*stride+window_size].clone()
                # window_i_m_1 = x[:,t-1*stride:t-1*stride+window_size ].clone()
                # window_i = x[:,t: t+window_size].clone()
                # window_i_p_1 = x[:,t+1*stride: t+1*stride+window_size].clone()
                # window_i_p_2 = x[:,t+2*stride: t+2*stride+window_size].clone()

                # w_1=compute_energy(window_i_m_2)
                
                # w_2=compute_energy(window_i_m_1)
                # w_3=compute_energy(window_i)
                # w_4=compute_energy(window_i_p_1)
                # w_5=compute_energy(window_i_p_2)
                # d_wi=(w_4-w_2)/((t+1*stride)-(t-1*stride))
                # d_wi_m_1=(w_3-w_1)/(t-(t-2*stride))
                # d_wi_p_1=(w_5-w_3)/((t+2*stride)-t)
                # if d_wi < eps and d_wi_m_1 > 0 and d_wi_p_1 < 0:
                compute_std(window,label_l)
                score = model(window)
                prob=torch.nn.functional.softmax(score, dim=-1)

                score_list_labels= torch.argmax(prob, dim=-1)
                # print(score_list_labels)
                # print(prob)
                # print(label)
                # input()
                if prob[0][score_list_labels[0].item()] < thresholds[str(score_list_labels[0].item())]['threshold_avg']:
                    score[0][0]=10.


                prob=torch.nn.functional.softmax(score, dim=-1)

                score_list_labels= torch.argmax(prob, dim=-1)

                # print("through")
                res.append({
                    "prediction":score_list_labels.item(),
                    "label":label.item(),
                })
                
                if score_list is None:
                    score_list = score
                    label_list = label
                    window_label_list=label_l.unsqueeze(0)
                    window_score_list=torch.cat([score_list_labels for _ in label_l]).unsqueeze(0)
                else:
                    score_list = torch.cat((score_list, score), 0)
                    label_list = torch.cat((label_list, label), 0)
                    window_label_list = torch.cat((window_label_list, label_l.unsqueeze(0)), 0)
                    window_score_list = torch.cat((window_score_list, torch.cat([score_list_labels for _ in label_l]).unsqueeze(0)), 0)


            results.append({
                "idx":i,
                "res":res
            })

            if cnf_matrix == None :
                cnf_matrix=confusion_matrix(score_list.detach().cpu(),label_list.detach().cpu(),num_classes=num_classes)
            else :
                cnf_matrix+=confusion_matrix(score_list.detach().cpu(),label_list.detach().cpu(),num_classes=num_classes)
            plot_confusion_matrix(cnf_matrix,labels,"./cnf_matrix.png",mode="png")
            plot_confusion_matrix(cnf_matrix,labels,"./cnf_matrix.eps",mode="eps")
            loss = criterion(score_list.detach().cpu(), label_list.detach().cpu())
            score_list_labels= torch.argmax(torch.nn.functional.softmax(score_list, dim=-1), dim=-1)
            val_f1_step= f1_score(score_list.detach().cpu(), label_list.detach().cpu())
            val_jaccard_step= get_jaccard(window_score_list.detach().cpu(), window_label_list.detach().cpu())
            val_fp_rate_step= get_fp_rate(score_list.detach().cpu(), label_list.detach().cpu())
            val_detection_rate=get_detection_rate(score_list.detach().cpu(), label_list.detach().cpu())
            val_avg_precision_step=avg_precision(score_list.detach().cpu(), label_list.detach().cpu())
            val_f1_epoch += val_f1_step
            val_jaccard_epoch += val_jaccard_step
            val_fp_rate_epoch += val_fp_rate_step
            val_avg_precision_epoch+=val_avg_precision_step
            val_det_rate_epoch+=val_detection_rate
            val_loss += loss
            print("*** SHREC  21"
                "val_loss_step: %.6f,"
                "val_F1_step: %.6f ***,"
                "val_jaccard_step: %.6f ***"
                "val_fp_rate_step: %.6f ***"
                "val_avg_precision_step: %.6f ***"
                "val_detection_rate_step: %.6f ***"
                % ( loss, val_f1_step,val_jaccard_step, val_fp_rate_step,val_avg_precision_step,val_detection_rate))
            with open('results_analysis.json',mode="w") as f:
                json.dump(results,f,indent=2)
        val_loss = val_loss / (float(i + 1))
        val_f1 = val_f1_epoch.item() / (float(i + 1))
        val_jaccard = val_jaccard_epoch / (float(i + 1))
        val_fp_rate = val_fp_rate_epoch / (float(i + 1))
        val_avg_precision = val_avg_precision_epoch / (float(i + 1))
        val_det_rate = val_det_rate_epoch / (float(i + 1))
        print("*** SHREC 21, "
                "val_loss: %.6f,"
                "val_F1: %.6f ***,"
                "val_jaccard: %.6f ***"
                "val_fp_rate: %.6f ***"
                "val_avg_precision_rate: %.6f ***"
                "val_detection_rate: %.6f ***"
                % (val_loss, val_f1,val_jaccard, val_fp_rate, val_avg_precision,val_det_rate))


