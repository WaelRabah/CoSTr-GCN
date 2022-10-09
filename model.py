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
from loss import FocalLoss
import json

 
class CoSTrGCN(pl.LightningModule):

    def __init__(self, adjacency_matrix,optimizer_params, labels, num_classes : int=14, d_model: int=512, n_heads: int=8,
                 nEncoderlayers: int=6, dropout: float = 0.1):
        super(CoSTrGCN, self).__init__()
        # not the best model...
        self.labels=labels
        features_in=3       
        self.cnf_matrix= torch.zeros(num_classes, num_classes).cuda()   
        self.Learning_Rate, self.betas, self.epsilon, self.weight_decay=optimizer_params
        self.num_classes=num_classes
        self.adjacency_matrix=adjacency_matrix.float()
        self.is_continual=False 
        self.threshold={
            i:{"count":0,"threshold_sum":.0,"threshold_avg":.0} for i in range(self.num_classes)
        }
        self.best_val_acc=.0
        self.current_val_acc=.0
        self.count=0
        self.loss=nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.confusion_matrix=torchmetrics.ConfusionMatrix(num_classes)
        self.gcn=SGCN(features_in,d_model,self.adjacency_matrix)

        self.encoder=TransformerGraphEncoder(is_continual=self.is_continual,dropout=dropout,num_heads=n_heads,dim_model=d_model, num_layers=nEncoderlayers)

        self.out = nn.Sequential(
            nn.Linear(d_model, d_model,dtype=torch.float).cuda(),
            nn.Mish(),
            nn.LayerNorm(d_model,dtype=torch.float).cuda(),
            nn.Linear(d_model,num_classes,dtype=torch.float).cuda()
          )

        self.d_model = d_model
        self.init_parameters()
    def init_parameters(self):
        for name,p in self.named_parameters() :
          if p.dim() > 1:
              nn.init.xavier_uniform_(p)
    def forward(self, x):
        x=x.type(torch.float).cuda() 
        
        #spatial features from SGCN
        x=self.gcn(x,self.adjacency_matrix)
        
        # temporal features from TGE
        x=self.encoder(x)

        # Global average pooling
        N,T,V,C=x.shape
        x=x.permute(0,3,1,2)
        # V pooling
        x = F.avg_pool2d(x, kernel_size=(1, V)).view(N,C,T)
        
        # T pooling
        x = F.avg_pool1d(x, kernel_size=T).view(N,C)
        
        # Classifier
        x=self.out(x)
        
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
        loss = self.loss(y_hat, y)

        #l1 regularization
        l1_lambda = 1e-4
        l1_norm = sum( p.abs().sum()  for p in self.parameters())

        loss_with_l1 = loss + l1_lambda * l1_norm

        self.train_acc(y_hat, y)
        
        self.log('train_loss', loss,on_epoch=True,on_step=True)
        self.log('train_acc', self.train_acc.compute(), prog_bar=True, on_step=True, on_epoch=True)



        return loss

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        
        x = batch[0].float()
        y = batch[1]
        y = y.type(torch.LongTensor)
        y = y.cuda()
        targets = Variable(y, requires_grad=False)
        y_hat = self(x)
        loss = self.loss(y_hat, targets)
        for y_e,gt in zip(y,y_hat) :
            if y_e==gt.argmax():
                prob=torch.nn.functional.softmax(gt, dim=-1)[y_e]
                self.threshold[y_e.item()]["threshold_sum"]+=prob.item()
                self.threshold[y_e.item()]["count"]+=1
            

                # print(self.threshold[y_e.item()])
        self.valid_acc(y_hat, y)
        self.current_val_acc+=self.valid_acc.compute()
        self.count+=1
        self.log('val_loss', loss, prog_bar=True,on_epoch=True,on_step=True)
        self.log('val_accuracy', self.valid_acc.compute(), prog_bar=True, on_step=True, on_epoch=True)



    def training_epoch_end(self, outputs):
        
        self.train_acc.reset()

    def validation_epoch_end(self, outputs):
        self.valid_acc.reset()
        print(self.current_val_acc / self.count)
        if (self.current_val_acc / self.count) > self.best_val_acc :
            self.best_val_acc=self.current_val_acc / self.count
            for k in self.threshold.keys():
                if self.threshold[k]["count"] > 0 :
                    self.threshold[k]["threshold_avg"]=self.threshold[k]["threshold_sum"] / self.threshold[k]["count"]
            with open('thresholds.json',mode="w") as f:
                json.dump(self.threshold,f,indent=2)
        self.threshold={
                        i:{"count":0,"threshold_sum":.0,"threshold_avg":.0} for i in range(self.num_classes)
                        }
        self.current_val_acc=.0
        self.count=0

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x = batch[0].float()
        y = batch[1]
        y = y.type(torch.LongTensor)
        y = y.cuda()
        targets = Variable(y, requires_grad=False)
        y_hat = self(x)
        _, preds = torch.max(y_hat, 1)
        self.test_acc(y_hat, targets)
        for y_e,gt in zip(y,y_hat) :
            if y_e==gt.argmax():
                prob=torch.nn.functional.softmax(gt, dim=-1)[y_e]
                self.threshold[y_e.item()]["threshold_sum"]+=prob.item()
                self.threshold[y_e.item()]["count"]+=1
        loss = self.loss(y_hat, targets)        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_accuracy', self.test_acc.compute(), prog_bar=True)


        
        self.cnf_matrix+=self.confusion_matrix(preds,targets)

    def on_test_end(self):
        time_now=datetime.today().strftime('%Y-%m-%d_%H_%M_%S')
        self.plot_confusion_matrix(f"./Confusion_matrices/Confusion_matrix_{time_now}.eps")

        for k in self.threshold.keys():
            if self.threshold[k]["count"] > 0 :
                self.threshold[k]["threshold_avg"]=self.threshold[k]["threshold_sum"] / self.threshold[k]["count"]
        with open('thresholds.json',mode="w") as f:
            json.dump(self.threshold,f,indent=2)


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