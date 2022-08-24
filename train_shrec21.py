import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import time
import os
import continual as co
from model import STrGCN
from data_loaders.shrec21_loader import load_data_sets
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
DATASETS_PATH = "datasets/"
DS_NAME = "shrec21"
DS_PATH = DATASETS_PATH + "shrec21/"
train_path = str(DS_PATH + "train_data_joint.npy")
test_path = str(DS_PATH + "test_data_joint.npy")
train_label = str(DS_PATH + "train_label.pkl")
test_label = str(DS_PATH + "test_label.pkl")
batch_size = 32
workers = 4
lr = 1e-3
hparams = {
    "forward_mode": "frame",
    "predict_after_frames": 0,
    "continual_temporal_fill": "replicate",
    "pool_size": -1,
    "pool_padding": -1,
    "profile_model":False
}
num_classes = 18
window_size=100
input_shape = (window_size,20,3)

device = torch.device('cuda')
# def compute_energy(x):
#     N, T, V, C = x.shape
#
#     x_values= x[:,:,:,0]
#     y_values = x[:, :, :, 1]
#     z_values = x[:, :, :, 2]
#     w=None
#     for v in range(V):
#         w_v=None
#         for t in range(1,T):
#             if w_v == None :
#                 w_v = torch.sqrt(( x_values[:,t,v]/x_values[:,t-1,v] -1)**2 + ( y_values[:,t,v]/y_values[:,t-1,v] -1)**2 + ( z_values[:,t,v]/z_values[:,t-1,v] -1)**2)
#             else :
#                 w_v  += torch.sqrt((x_values[:, t, v] / x_values[:, t - 1, v] - 1) ** 2 + (
#                             y_values[:, t, v] / y_values[:, t - 1, v] - 1) ** 2 + (
#                                            z_values[:, t, v] / z_values[:, t - 1, v] - 1) ** 2)
#         if w==None :
#             w=w_v
#         else :
#             w+=w_v
def init_data_loader():
    train_dataset, test_dataset, graph= load_data_sets()
    print("train data num: ", len(train_dataset))
    print("test data num: ", len(test_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=False)

    return train_loader, val_loader, graph


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
    model_fold = "./models/cost_gcn/online_model_checkpoints/"
    try:
        os.mkdir(model_fold)
    except:
        pass

    train_loader, val_loader, graph = init_data_loader()

    # .........inital model
    print("\ninit model.............")
    model = init_model(graph, num_classes,300)
    model_solver = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # ........set loss
    criterion = torch.nn.CrossEntropyLoss()

    #
    train_data_num = 1960
    test_data_num = 840
    iter_per_epoch = int(train_data_num / batch_size)

    # parameters recording training log
    max_acc = 0
    no_improve_epoch = 0

    n_iter = 0
    epochs = 500
    f1_score=torchmetrics.F1(num_classes=num_classes)
    # It says IoU but in fact IoU and Jaccard index have the same formula
    jaccard = torchmetrics.IoU(num_classes=num_classes)

    # ***********training#***********
    for epoch in tqdm(range(epochs)):
        print("\ntraining.............")
        model.train()
        start_time = time.time()
        train_F1 = 0
        train_jaccard = 0
        train_fp_rate = 0
        train_loss = 0
        score_list = None
        label_list = None
        print("Epoch=", epoch)
        for i, batch in tqdm(enumerate(train_loader), leave=False):
            n_iter += 1
            # print("training i:",i)
            if i + 1 > iter_per_epoch:
                continue
            x, target = batch[0], batch[1]

            N, T, V, C = x.shape
            target = torch.stack([torch.stack([target[j][i] for j in range(T)]) for i in range(N)])
            # x = x.permute(0, 3, 1, 2)
            preds = []
            acc_sum = .0
            loss_sum = .0
            num_windows= T // window_size
            for w in range(num_windows):
                current_skeletons_window = x[:,w*window_size: (w+1)*window_size].clone()
                # print(current_skeletons_window.shape)
            # for t in tqdm(range(T), leave=False):
            #     current_skeleton = x[:, :, t].clone().unsqueeze(2)

                score = model(current_skeletons_window)
                if type(score) == co.module.TensorPlaceholder:

                    continue
                # preds.append((pred, ))

                label = target[:, w*window_size: (w+1)*window_size].cuda()
                label = get_window_label(label)
                label = torch.autograd.Variable(label, requires_grad=False)
                # print(score.shape, label.shape)





                if score_list is None:
                    score_list = score
                    label_list = label
                else:
                    score_list = torch.cat((score_list, score), 0)
                    label_list = torch.cat((label_list, label), 0)
                # acc_sum += acc
            loss = criterion(score_list.cuda(), label_list.cuda())
            model.zero_grad()
            loss.backward()
            # clip_grad_norm_(model.parameters(), 0.1)
            model_solver.step()
            score_list = torch.argmax(torch.nn.functional.softmax(score_list, dim=-1), dim=-1)
            train_F1 += f1_score(score_list.detach().cpu(),label_list.detach().cpu())
            train_jaccard += jaccard(score_list.detach().cpu(), label_list.detach().cpu())
            train_fp_rate+=get_fp_rate(score_list, label_list)
            train_loss += loss

        train_F1 /= float(i + 1)
        train_loss /= float(i + 1)
        train_jaccard /= float(i + 1)
        train_fp_rate /= float(i + 1)
        # print(train_fp_rate)
        print("*** SHREC  Epoch: [%2d] time: %4.4f, "
              "cls_loss: %.4f  train_F1: %.6f train_jaccard %.6f train_fp_rate %.6f ***"
              % (epoch + 1, time.time() - start_time,
                 train_loss.data, train_F1,train_jaccard, train_fp_rate))
        start_time = time.time()

            # ***********evaluation***********
        with torch.no_grad():
            val_loss = 0
            val_cc = 0
            val_jaccard=0
            val_fp_rate=0
            score_list = None
            label_list = None
            acc_sum = 0
            model.eval()
            for i, batch in tqdm(enumerate(train_loader), leave=False):
                n_iter += 1
                # print("training i:",i)
                if i + 1 > iter_per_epoch:
                    continue
                x, target = batch[0], batch[1]

                N, T, V, C = x.shape
                target = torch.stack([torch.stack([target[j][i] for j in range(T)]) for i in range(N)])
                x = x.permute(0, 3, 1, 2)
                val_loss_batch = 0
                val_jaccard_batch=0
                val_fp_rate_batch=0
                val_acc_batch = 0
                for t in range(T):
                    current_skeleton = x[:, :, t].clone().unsqueeze(2)

                    score = model(current_skeleton).cuda()
                    # print(type(pred),t)
                    if type(score) == co.module.TensorPlaceholder:
                        continue
                    # preds.append((pred, ))

                    label = target[:, t].cuda()
                    label = torch.autograd.Variable(label, requires_grad=False)
                    # print(score.shape, label)
                    loss = criterion(score, label)
                    val_loss_batch += loss
                    score = torch.argmax(torch.nn.functional.softmax(score,dim=-1),dim=-1)
                    if score_list is None:
                        score_list = score
                        label_list = label
                    else:
                        score_list = torch.cat((score_list, score), 0)
                        label_list = torch.cat((label_list, label), 0)

                val_acc_batch += f1_score(score_list.detach().cpu(), label_list.detach().cpu())
                val_jaccard_batch += jaccard(score_list.detach().cpu(), label_list.detach().cpu())
                val_fp_rate_batch += get_fp_rate(score_list.detach().cpu(), label_list.detach().cpu())
                val_loss += val_loss_batch / T

            val_loss = val_loss / float(i + 1)
            val_cc = val_acc_batch.item() / float(i + 1)
            val_jaccard = val_jaccard_batch / float(i + 1)
            val_fp_rate = val_fp_rate_batch / float(i + 1)
            print("*** SHREC  Epoch: [%2d], "
                  "val_loss: %.6f,"
                  "val_F1: %.6f ***,"
                  "val_jaccard: %.6f ***"
                  "val_fp_rate: %.6f ***"
                  % (epoch + 1, val_loss, val_cc,val_jaccard, val_fp_rate))

            # save best model
            if val_cc > max_acc:
                max_acc = val_cc
                no_improve_epoch = 0
                val_cc = round(val_cc, 10)

                torch.save(model.state_dict(),
                           '{}/epoch_{}_acc_{}.pth'.format(model_fold, epoch + 1, val_cc))
                print("performance improve, saved the new model......best F1 score: {}".format(max_acc))
            else:
                no_improve_epoch += 1
                print("no_improve_epoch: {} best score {}".format(no_improve_epoch, max_acc))

            if no_improve_epoch > 10:
                print("stop training....")
                break
