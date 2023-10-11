import time, os, sys, random, string
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch import nn
from matplotlib import cm
from logger import Logger
from sklearn.manifold import TSNE
from torchmetrics import AUROC, Accuracy, Precision, Specificity, ConfusionMatrix, F1Score
from sklearn.linear_model import LogisticRegression
from models.encoders import *



# from evaluation.evaluate_generator_coco import calculate_metrics_coco

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from get_data import get_loader
from data_scripts.utils import  values_target, weights_init,vectors_to_images, custom_grad_hook


class Experiment:
    """ The class that contains the experiment details """
    def __init__(self, experimentType):
        """
        Standard init
        :param experimentType: experiment enum that contains all the data needed to run the experiment
        :type experimentType:
        """

        #Extract Paramters from Experiment Enum
        self.name = experimentType.name
        self.type = experimentType.value

        self.encoder_emb_sz = self.type["encoder_emb_sz"] #Hyper-paramter (encoder output/ generator input)
        self.modalities=self.type["modalities"]
        self.isMultiModal=self.type["isMultiModal"]
        self.selected_features=self.type["selected_features"]
        self.isContrastive=self.type["isContrastive"]
        self.useAI_Segmentation=self.type["useAI_Segmentation"]
        self.biasFeildCorrection=self.type["biasFeildCorrection"]
        self.toTrain=self.type["toTrain"]
        self.resample = self.type["resample"]
        self.targetVoxelShape=self.type["targetVoxelShape"]
        self.targetimageSize=self.type["targetimageSize"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cuda = True if torch.cuda.is_available() else False
        self.batchSize=self.type["batchSize"]
        
        self.encoder_weights_path = self.type["encoder_weights_path"]
        if self.encoder_weights_path is None:
            self.encoder = self.type["encoder"]().to(self.device)
        else:
            self.encoder = EncoderNet_pretrained_contrastive(self.type["encoder"], self.encoder_weights_path)
        
        self.augmentation = self.type["augmentation"]
        self.table_corruption = self.type["table_corruption"]
        self.encoder_optim = self.type["encoder_optim"](self.encoder.parameters(), lr=self.type["encoder_lr"], betas=(0.5, 0.99))
        self.encoderTab= self.type["tabular_encoder"](n_features=len(self.selected_features)).to(self.device)
        self.embClassifer = None if self.type["embClassifer"] is None else self.type["embClassifer"]().to(self.device)
        self.loss = self.type["loss"]
        self.accuracy = Accuracy(task="binary").to(self.device)
        self.epochs = self.type["epochs"]
        self.batchAccumulator=self.type["batchAccumulator"] #accumlerator the updates for small batches
        self.avaliable_update=False #falg to indicate a missing optimizing step.

        torch.backends.cudnn.benchmark = True

    def run(self, logging_frequency=4):
        """
        This function runs the experiment
        :param logging_frequency: how frequently to log each epoch (default 4)
        :type logging_frequency: int
        :return: None
        :rtype: None
        """
        
        start_time = time.time()

        logger = Logger(self.name, self.type["dataset"])
        logger.save_hparams(self.type)

        # calculate_metrics_coco(sampling_args,numberOfSamples=15)
        # return

        # self.encoder.apply(weights_init)

        train_loader,val_loader,test_loader ,(train_GT,val_GT,test_GT) = get_loader( 
               batchSize=self.type["batchSize"], toShuffle=self.type["toShuffle"], percentage=self.type["percentage"],
               dataset=self.type["dataset"], modalities=self.modalities, augmentation=self.augmentation,
               multiModal=self.isMultiModal,useAI_Segmentation=self.useAI_Segmentation,biasFeildCorrection=self.biasFeildCorrection,
               resample=self.resample,targetVoxelShape=self.targetVoxelShape,targetimageSize = self.targetimageSize,
               selected_features=self.selected_features, corruption = self.table_corruption,
               )
        num_batches = len(train_loader)

        if self.cuda:
            self.encoder = self.encoder.cuda()
            self.loss = self.loss.cuda()

        # track losses
        best_train_error,best_val_error,best_val_AUROC=10000,10000,0
        batch_scan,batch_aug_scan,batch_table_feat,batch_case_csPCa=None,None,None,None
        train_losses,train_accuracies = [],[]
        epoch_val_losses,epoch_val_accuracies,epoch_val_AUROC =[],[],[]
        embeddings,labels=[],[] #for online classification in contrastive learning
        estimator=None 
        #logger.save_GT(val_GT,name="Val_GT")
        #logger.save_GT(test_GT,name="Test_GT")
        
        # Train model
        if self.toTrain:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.encoder_optim, self.epochs)
            for epoch in range(1, self.epochs + 1):
                self.encoder.train()
                # self.encoder.out.register_backward_hook(custom_grad_hook) #use custom grad hook if needed
                for n_batch, (batch_data) in enumerate(train_loader):
                    batch_scan,batch_aug_scan,batch_table_feat,batch_case_csPCa = self._processBatch(batch_data)
                    
                    # Train Encoder
                    if self.isContrastive:
                        train_loss,train_acc,embedding = self._train_encoder_contrastive(batch_scan,batch_case_csPCa,batch_aug_scan,batch_table_feat,n_batch)
                        embeddings.append(embedding) #should we deatch from cuda here?
                        labels.append(batch_case_csPCa) #should we deatch from cuda here?
                    else:
                        train_loss,train_acc = self._train_encoder(batch_scan,batch_case_csPCa,batch_aug_scan,batch_table_feat,n_batch)

                    # Save Losses for plotting later
                    train_losses.append(train_loss)
                    train_accuracies.append(train_acc)
                    
                    logger.log(train_loss, epoch, n_batch, num_batches)

                    # Display status Logs
                    if n_batch % (num_batches // logging_frequency) == 0:
                        logger.display_status(
                            epoch, self.epochs, n_batch, num_batches,
                            train_loss,train_acc
                        )
            

                if self.isContrastive:
                    embeddings_hat = embeddings[0]
                    labels_hat = labels[0]
                    for i in range(1,len(embeddings)): #forgot why i did this, check if necessary
                        labels_hat = torch.cat((labels_hat.cuda(), labels[i].cuda()), dim=0) # have to be on same device??
                        embeddings_hat = torch.cat((embeddings_hat.cuda(), embeddings[i].cuda()), dim=0)
                        embeddings_hat = embeddings_hat.detach().cpu()
                        labels_hat = labels_hat.cpu()
                
                    #get logistic regression model
                    estimator = LogisticRegression(class_weight='balanced', max_iter=1000).fit(embeddings_hat, labels_hat)
                    embeddings.clear()
                    labels.clear()
                    val_loss,val_acc,val_AUROC = self._testModel_contrastive(estimator,val_loader,logger=logger)
                    scheduler.step()
                else:
                    val_loss,val_acc,val_AUROC = self._testModel(val_loader,logger=logger)
            
            
                epoch_val_losses.append(val_loss)
                epoch_val_accuracies.append(val_acc)
                epoch_val_AUROC.append(val_AUROC)
                logger.display_Valdiation_status(epoch,self.epochs, val_loss, val_acc, val_AUROC)
            
                # Save better model
                if(val_AUROC>=best_val_AUROC):
                    logger.save_model(model=self.encoder,name="encoder",epoch=epoch,loss=val_loss,AUROC=val_AUROC)
                    best_val_AUROC=val_AUROC
                
            if self.avaliable_update: # one final optimizing step is missing 
                self.encoder_optim.step() # update parameters
                self.encoder_optim.zero_grad() # Reset gradients
                self.avaliable_update=False
            
            logger.save_errors(train_loss=train_losses,val_loss=epoch_val_losses,val_AUROC=epoch_val_AUROC,num_batches=num_batches)            
            logger.save_predictions(predictions=None,name="Val_predictions")
        
        timeTaken = time.time() - start_time
                   
        # Test model on Validation
        self.encoder=logger.load_model(model=self.encoder,name="encoder") #load best weights
        if self.isContrastive:
            if self.embClassifer is None: self.embClassifer = estimator
            # test_loss,test_acc,test_AUROC = self._testModel_contrastive(self.embClassifer,test_loader,logger=logger)
            test_loss,test_acc,test_AUROC = self._testModel_contrastive(self.embClassifer,val_loader,logger=logger)

        else:
            # test_loss,test_acc,test_AUROC = self._testModel(test_loader,logger=logger)
            test_loss,test_acc,test_AUROC = self._testModel(val_loader,logger=logger)

        logger.save_predictions(predictions=None,name="Test_predictions_onValdSet")
        model_classification_report = self.confusion_matrix(val_GT,"Test_predictions_onValdSet",logger,name="test_onValdSet")
        logger.save_scores(timeTaken,test_acc,test_AUROC,model_classification_report)
        return

    def _processBatch(self,batch_data):
        batch_scan,batch_aug_scan,batch_table_feat,batch_case_csPCa=None,None,None,None
        if  self.type["dataset"]=='picai' or self.type["dataset"]=='Balanced-picai' :
            if self.isMultiModal:
                batch_scan, batch_aug_scan,batch_case_csPCa,batch_table_feat = batch_data
                batch_table_feat=torch.stack(batch_table_feat)
                N = batch_aug_scan.shape[0]
                if self.cuda:
                    batch_scan = batch_scan.cuda()
                    batch_aug_scan = batch_aug_scan.cuda()
                    batch_case_csPCa = batch_case_csPCa.cuda().to(torch.float32)
                    batch_table_feat = torch.transpose(batch_table_feat, 0, 1)
                    batch_table_feat = torch.nan_to_num(batch_table_feat)
                    batch_table_feat = batch_table_feat.cuda().to(torch.float32)
            else:
                batch_scan, batch_aug_scan, batch_case_csPCa = batch_data
                N = batch_scan.shape[0]
                if self.cuda:
                    batch_scan = batch_scan.cuda()
                    batch_aug_scan = batch_aug_scan.cuda()
                    batch_case_csPCa = batch_case_csPCa.cuda().to(torch.float32)
        
        else : #other datasets
            batch_scan, batch_case_csPCa = batch_data
            N = batch_scan.size(0)
            if self.cuda:
                batch_scan = batch_scan.cuda()
                batch_case_csPCa = batch_case_csPCa.cuda()

        return batch_scan,batch_aug_scan,batch_table_feat,batch_case_csPCa

    def _train_encoder(self, batch_scan: torch.Tensor,batch_case_csPCa: torch.Tensor,
                       batch_aug_scan: torch.Tensor,batch_table_feat: torch.Tensor,n_batch) -> torch.Tensor:
        """
        This function performs one iteration of training 
        """

        N = batch_scan.size(0)
        y_hat = self.encoder(batch_aug_scan).view(-1) 
        y_hat = nn.functional.sigmoid(y_hat)

        # Calculate error and back-propagate
        error = self.loss(y_hat,batch_case_csPCa)
        accuracy = (y_hat.round() == batch_case_csPCa).float().mean()

        error=error/self.batchAccumulator
        accuracy=accuracy/self.batchAccumulator
        error.backward()
        self.avaliable_update=True

        # clip gradients to avoid exploding gradient problem
        # nn.utils.clip_grad_norm_(self.encoder.parameters(), 10)

        if ((n_batch+1)%self.batchAccumulator) ==0:
            self.encoder_optim.step() # update parameters
            self.encoder_optim.zero_grad() # Reset gradients
            self.avaliable_update=False

        # Return error
        return error.item(),accuracy.item()
       
    def _train_encoder_contrastive(self, batch_scan: torch.Tensor,batch_case_csPCa: torch.Tensor,
                       batch_aug_scan: torch.Tensor,batch_table_feat: torch.Tensor,n_batch) -> torch.Tensor:
        """
        This function performs one iteration of training the encoder contrastively
        """

        N = batch_scan.size(0)

        
        if self.isMultiModal:
            y0_hat,_ = self.encoder(batch_aug_scan)#.view(-1) #todo: in Multimodal, should we use scan or aug scan?
            y1_hat,_ = self.encoderTab(batch_table_feat)
        else:
            y0_hat,_ = self.encoder(batch_scan)#.view(-1)
            y1_hat,_ = self.encoder(batch_aug_scan)#.view(-1)

        # Calculate error and back-propagate
        if self.isMultiModal:
            error, _,_= self.loss(y0_hat,y1_hat)
        else:
            error = self.loss(y0_hat,y1_hat)
        #accuracy = (y_hat.round() == batch_case_csPCa).float().mean()

        error=error/self.batchAccumulator
        #accuracy=accuracy/self.batchAccumulator
        error.backward()
        self.avaliable_update=True

        if ((n_batch+1)%self.batchAccumulator) ==0:
            self.encoder_optim.step() # update parameters
            self.encoder_optim.zero_grad() # Reset gradients
            self.avaliable_update=False

        return error.item(),0, y0_hat

    def _testModel(self,testload,logger=None):
        self.encoder.eval()
        test_losses,test_pred,test_preds_proba, test_labels=[],[],[],[]
        batch_scan,batch_aug_scan,batch_table_feat,batch_case_csPCa=None,None,None,None
        for n_batch, (batch_data) in enumerate(testload):
            batch_scan,batch_aug_scan,batch_table_feat,batch_case_csPCa = self._processBatch(batch_data)
            
            #forwad pass
            N = batch_scan.size(0) 
            y_hat = self.encoder(batch_scan).view(-1) #generate a new fake image to train the Generator & Encoder
            y_hat = nn.functional.sigmoid(y_hat)

            # Calculate error & accuracy
            loss = self.loss(y_hat,batch_case_csPCa)
            preds=y_hat.round()
            test_preds_proba.extend(y_hat.detach().cpu())
            test_losses.append(loss.item())
            test_pred.extend(preds.detach().cpu().numpy())
            test_labels.extend(batch_case_csPCa.detach().cpu().numpy())
            if logger: logger.save_predictions(preds)
        
        #Cal performance mertircs 
        mertic_Acc=Accuracy(task="binary")
        test_acc= mertic_Acc(torch.tensor(test_pred),torch.tensor(test_labels))
        
        
        mertic_Acc = AUROC(task="binary") 
        metric_AUROC = mertic_Acc(torch.tensor(test_preds_proba),torch.tensor(test_labels))
        
        print(f'AUROC = {metric_AUROC},Accuracy = {test_acc.item()}')
        return np.mean(test_losses),test_acc.item(),metric_AUROC.item()

    def _testModel_contrastive(self, estimator, testload,logger=None):
        self.encoder.eval()
        test_losses,test_pred,test_preds_proba, test_labels=[],[],[],[]
        batch_scan,batch_aug_scan,batch_table_feat,batch_case_csPCa=None,None,None,None
        t1=0
        t5=0
        for n_batch, (batch_data) in enumerate(testload):
            batch_scan,batch_aug_scan,batch_table_feat,batch_case_csPCa = self._processBatch(batch_data)
            with torch.no_grad():   
                y0_hat,h0 = self.encoder(batch_scan)#.view(-1)
                if self.isMultiModal:
                    y1_hat,h1 = self.encoderTab(batch_table_feat)
                else:
                    y1_hat,h1 = self.encoder(batch_aug_scan)#.view(-1)
            if self.isMultiModal:
                test_loss,logit,label = self.loss(y0_hat,y1_hat)
                top1_acc_val = Accuracy(task='multiclass', top_k=1,num_classes=label.size(0)).cpu()
                top5_acc_val = Accuracy(task='multiclass', top_k=5,num_classes=label.size(0)).cpu()
                top1=top1_acc_val(logit.cpu(), label.cpu())
                top5=top5_acc_val(logit.cpu(), label.cpu())
                t1=t1+top1
                t5=t5+top5
            else:
                test_loss = self.loss(y0_hat,y1_hat)

            
            preds = estimator.predict(y0_hat.detach().cpu().numpy())
            preds_proba = estimator.predict_proba(y0_hat.detach().cpu().numpy())
            test_preds_proba.extend(preds_proba[:,1])
            test_pred.extend(preds)
            
            test_losses.append(test_loss.item())
            test_labels.extend(batch_case_csPCa.detach().cpu().numpy())
            
        
        #Cal performance mertrics 
        mertic = AUROC(task="binary") 
        metric_AUROC = mertic(torch.tensor(test_preds_proba),torch.tensor(test_labels))

        mertic_Acc=Accuracy(task="binary")
        test_acc=mertic_Acc(torch.tensor(test_pred),torch.tensor(test_labels))
        #mertic_Acc=Precision(task="binary")
        #test_pre=mertic_Acc(torch.tensor(test_pred),torch.tensor(test_labels))
        #mertic_Acc=Specificity(task="binary")
        #test_spe=mertic_Acc(torch.tensor(test_pred),torch.tensor(test_labels))

        #print(f'top1 = {t1/3},top5= {t5/3}')
        #print(f'AUROC = {metric_AUROC}')#,Precision = {test_pre},Specificity = {test_spe},Accuracy = {test_acc},')

        if logger: logger.save_predictions(test_pred)
        return np.mean(test_losses),test_acc.item(),metric_AUROC.item()

    def confusion_matrix(self,labels, predicitions_filename,logger,name=None,classes=None):
        from sklearn.metrics import classification_report
        
        if not logger.toEvaluate:
            path=f'{logger.data_subdir}/{predicitions_filename}.npy'
        else:
            path=f'{logger.eval_data_subdir}/{predicitions_filename}.npy'

        predicitions = np.load(path)
        self.plot_confusion_matrix(labels,predicitions,logger,name=name)
        model_classification_report = classification_report(labels,predicitions,digits=4)
        print(model_classification_report)
        return model_classification_report

    def plot_confusion_matrix(self,labels, predicitions,logger,name=None,classes=None):
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import classification_report
        from sklearn.metrics import ConfusionMatrixDisplay
        from sklearn.metrics import multilabel_confusion_matrix

        fig = plt.figure(figsize = (5, 5));
        ax = fig.add_subplot(1, 1, 1);
        cm = confusion_matrix(labels, predicitions);
        cm = ConfusionMatrixDisplay(cm, display_labels = classes);
        cm.plot(values_format = 'd', cmap = 'Blues', ax = ax)
        plt.xticks(rotation = 20)
        if name is not None: name=name+"_confusionMatrix"
        else:  name = "confusionMatrix"
        logger.savefig(fig,name)

    def evaluate(self):
        logger = Logger(self.name, self.type["dataset"],toEvaluate= True)
        _,_,test_loader ,(_,_,test_GT) = get_loader( 
               batchSize=self.type["batchSize"], toShuffle=self.type["toShuffle"], percentage=1,
               dataset=self.type["dataset"], modalities=self.modalities, augmentation=self.augmentation,
               multiModal=self.isMultiModal,useAI_Segmentation=self.useAI_Segmentation,biasFeildCorrection=self.biasFeildCorrection,
               resample=self.resample,targetVoxelShape=self.targetVoxelShape,targetimageSize = self.targetimageSize,
               selected_features=self.selected_features, corruption = self.table_corruption,
               )
        num_batches = len(test_loader)

        if self.cuda:
            self.encoder = self.encoder.cuda()
            self.loss = self.loss.cuda()
        
        # Test model on Validation
        self.encoder=logger.load_model(model=self.encoder,name="encoder") #load best weights

        start_time = time.time()
        if self.isContrastive:
            test_loss,test_acc,test_AUROC = self._testModel_contrastive(self.embClassifer,test_loader,logger=logger)
        else:
            test_loss,test_acc,test_AUROC = self._testModel(test_loader,logger=logger)
        
        timeTaken = time.time() - start_time
        logger.save_predictions(predictions=None,name="Test_predictions_onTestSet")
        model_classification_report = self.confusion_matrix(test_GT,"Test_predictions_onTestSet",logger,name="test_onTestSet")
        logger.save_scores(timeTaken,test_acc,test_AUROC,model_classification_report)