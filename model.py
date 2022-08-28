import torch
from layers.continual_transformer_layers import  TransformerGraphEncoder
from layers.SGCN import  SGCN
import torch.nn as nn
import pytorch_lightning as pl
from torch.autograd import Variable
from torch.nn import functional as F
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
# import torch_optimizer as optim
# model definition



 
class STrGCN(pl.LightningModule):

    def __init__(self, adjacency_matrix,optimizer_params, labels, num_classes : int=18, d_model: int=512, n_heads: int=8,
                 nEncoderlayers: int=6, dropout: float = 0.1):
        super(STrGCN, self).__init__()
        # not the best model...
        self.labels=labels
        features_in=3       
        self.cnf_matrix= torch.zeros(num_classes, num_classes).cuda()
        self.Learning_Rate, self.betas, self.epsilon, self.weight_decay=optimizer_params
        self.num_classes=num_classes
        self.adjacency_matrix=adjacency_matrix.float()
        self.is_continual=False
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.val_f1_score=torchmetrics.F1Score(num_classes)
        self.train_f1_score=torchmetrics.F1Score(num_classes)
        self.test_f1_score=torchmetrics.F1Score(num_classes)
        self.val_jaccard=torchmetrics.JaccardIndex(num_classes)
        self.train_jaccard=torchmetrics.JaccardIndex(num_classes)
        self.test_jaccard=torchmetrics.JaccardIndex(num_classes)
        self.confusion_matrix=torchmetrics.ConfusionMatrix(num_classes)
        self.gcn=SGCN(features_in,d_model,self.adjacency_matrix)

        self.encoder=TransformerGraphEncoder(is_continual=self.is_continual,dropout=dropout,num_heads=n_heads,dim_model=d_model, num_layers=nEncoderlayers)

        self.out = nn.Sequential(
            nn.Linear(d_model, d_model,dtype=torch.float).cuda(),
            nn.Mish(),
            # nn.Dropout(dropout),
            nn.LayerNorm(d_model,dtype=torch.float).cuda(),
            nn.Linear(d_model,num_classes,dtype=torch.float).cuda()
          )

        self.d_model = d_model
        self.init_parameters()
    def init_parameters(self):
        for name,p in self.named_parameters() :
          if p.dim() > 1:
              nn.init.xavier_uniform_(p)
    def get_fp_rate(self,score,labels):
        


        cnf_matrix = self.confusion_matrix(score.detach().cpu(), labels.detach().cpu())
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
    def forward(self, x):
        # print(x.shape)
        x=x.type(torch.float).cuda() 
        
        # print(x.shape)
        #spatial features from SGCN
        x=self.gcn(x,self.adjacency_matrix)
        
        # print(x.shape)
        # print(x.shape)
        # temporal features from TGE
        x=self.encoder(x)
        
        
        # print(x.shape)

        # Global average pooling
        N,T,V,C=x.shape
        x=x.permute(0,3,1,2)
        # V pooling
        x = F.avg_pool2d(x, kernel_size=(1, V)).view(N,C,T)
        
        # T pooling
        x = F.avg_pool1d(x, kernel_size=T).view(N,C)
        
        # print(x)
        # Classifier
        x=self.out(x)
        # print(torch.equal(x[0],x[1]))
        
        return x
    def plot_confusion_matrix(self,filename,eps=1e-5) :
        import seaborn as sn
        confusion_matrix_sum_vec= torch.sum(self.cnf_matrix,dim=1) +eps
        
        confusion_matrix_percentage=(self.cnf_matrix /  confusion_matrix_sum_vec.view(-1,1) )

        plt.figure(figsize = (18,16))
        sn.heatmap(confusion_matrix_percentage.cpu().numpy(), annot=True,cmap="coolwarm", xticklabels=self.labels,yticklabels=self.labels)
        plt.savefig(filename,format="eps")
    def training_step(self, batch, batch_nb):
        # REQUIRED
        x = batch[0].float()
        y = batch[1]
        y = y.type(torch.LongTensor)
        y = y.cuda()
        y = Variable(y, requires_grad=False)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        # print(loss)
        # input()
        #l1 regularization
        l1_lambda = 1e-4
        l1_norm = sum( p.abs().sum()  for p in self.parameters())

        loss_with_l1 = loss + l1_lambda * l1_norm

        self.train_acc(y_hat, y)
        self.train_f1_score(y_hat, y)
        
        self.log('train_loss', loss,on_epoch=True,on_step=True)
        self.log('train_acc', self.train_acc.compute(), prog_bar=True, on_step=True, on_epoch=True)

        # self.log('train_F1_score', self.train_f1_score.compute(), prog_bar=True, on_step=True, on_epoch=True)
        # self.log('train_Jaccard', self.train_jaccard(y_hat, y), prog_bar=True, on_step=True, on_epoch=True)
        # self.log('train_FP_rate', self.get_fp_rate(torch.argmax(torch.nn.functional.softmax(y_hat, dim=-1), dim=-1), y), prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        
        x = batch[0].float()
        y = batch[1]
        y = y.type(torch.LongTensor)
        y = y.cuda()
        targets = Variable(y, requires_grad=False)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, targets)
        # print(loss)
        # input()
        self.valid_acc(y_hat, y)
        self.val_f1_score(y_hat, y)
        
        self.log('val_loss', loss, prog_bar=True,on_epoch=True,on_step=True)
        self.log('val_accuracy', self.valid_acc.compute(), prog_bar=True, on_step=True, on_epoch=True)

        # self.log('val_F1_score', self.val_f1_score.compute(), prog_bar=True, on_step=True, on_epoch=True)
        # self.log('val_Jaccard', self.val_jaccard(y_hat, y), prog_bar=True, on_step=True, on_epoch=True)
        # self.log('val_FP_rate', self.get_fp_rate(torch.argmax(torch.nn.functional.softmax(y_hat, dim=-1), dim=-1), y), prog_bar=True, on_step=True, on_epoch=True)


    def training_epoch_end(self, outputs):
        #for name,p in self.named_parameters() :
        #    print(p.shape)
        
        self.train_acc.reset()

    def validation_epoch_end(self, outputs):
        self.valid_acc.reset()

    def test_step(self, batch, batch_nb):
        # global confusion_matrix
        # OPTIONAL
        x = batch[0].float()
        y = batch[1]
        y = y.type(torch.LongTensor)
        y = y.cuda()
        targets = Variable(y, requires_grad=False)
        y_hat = self(x)
        _, preds = torch.max(y_hat, 1)
        self.test_acc(y_hat, targets)
        self.test_f1_score(y_hat, y)
        
        loss = F.cross_entropy(y_hat, targets)        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_accuracy', self.test_acc.compute(), prog_bar=True)

        # self.log('test_F1_score', self.val_f1_score.compute(), prog_bar=True)
        # self.log('test_Jaccard', self.test_jaccard(y_hat, y), prog_bar=True, on_step=True, on_epoch=True)
        # self.log('test_FP_rate', self.get_fp_rate(torch.argmax(torch.nn.functional.softmax(y_hat, dim=-1), dim=-1), y), prog_bar=True, on_step=True, on_epoch=True)

        
        self.cnf_matrix+=self.confusion_matrix(preds,targets)

    def on_test_end(self):
        time_now=datetime.today().strftime('%Y-%m-%d_%H_%M_%S')
        self.plot_confusion_matrix(f"./Confusion_matrices/Confusion_matrix_{time_now}.eps")

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        

        opt = torch.optim.RAdam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.Learning_Rate, weight_decay=self.weight_decay)
        reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode='min',
            factor=.5,
            patience=2,
            min_lr=1e-4,
            verbose=True
        )

        return  {"optimizer": opt, "lr_scheduler": reduce_lr_on_plateau, "monitor": "val_loss"}