import torch
import errno
import os
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
import numpy as np
import torchvision.utils as vutils
import json


class Logger:
    """
    Logs information during the experiment
    """
    def __init__(self, experiment_name, datasetName, toEvaluate=False):
        """
        Standard init
        :param experiment_name: name of the experiment enum
        :type experiment_name: str
        :param datasetName: name of the dataset
        :type datasetName: str
        """
        
        self.model_name = experiment_name
        self.data_subdir = f'./results/{datasetName}/{experiment_name}'
        self.eval_data_subdir = f'./evaluate/{datasetName}/{experiment_name}'
        self.server_data_subdir = f'/data1/practical-sose23/morphometric/results_arno_new/{datasetName}/{experiment_name}'
        self.toEvaluate = toEvaluate
        
        self.Validation_predictions=[]
        Logger._make_dir(self.data_subdir)
        if self.toEvaluate: Logger._make_dir(self.eval_data_subdir)

        # TensorBoard
        self.writer = SummaryWriter(comment=self.data_subdir, write_to_disk=False)

    def log(self, encoder_error, epoch, n_batch, num_batches):
        if isinstance(encoder_error, torch.autograd.Variable):
            encoder_error = encoder_error.data.cpu().numpy()

        step = Logger._step(epoch, n_batch, num_batches)
        self.writer.add_scalar(
            '{}/encoder_error'.format(self.data_subdir), encoder_error, step)
    
    def save_hparams(self, hparams):
        def convert_to_string(obj):
            # If the object is already a string, return it
            if isinstance(obj, str):
                return obj

            # If the object is a dictionary, recursively convert its values to strings
            if isinstance(obj, dict):
                return {key: convert_to_string(value) for key, value in obj.items()}

            # Convert the object to its string representation
            return str(obj)
        
        print(hparams, type(hparams))
        json_obj = json.dumps(convert_to_string(hparams), indent=4)

        with open(self.data_subdir + "/hparams.json", "w") as outfile:
            outfile.write(json_obj)

    def save_GT(self, GT, name="GT"): 
        if GT is None: return
        if isinstance(GT, torch.autograd.Variable):
            GT = GT.data.cpu().numpy()
        np.save(f'{self.data_subdir}/{name}.npy', np.array(GT))

    def save_predictions(self, predictions=None, name="predictions"):
        
        #append current epoch's predicitions
        if predictions is not None:      
            if isinstance(predictions, torch.autograd.Variable):
                predictions = predictions.data.cpu().numpy()
            self.Validation_predictions.extend(predictions)
        
        #or, save all predicitions of length: epochs x predicitions
        else:  
            if not self.toEvaluate :
                path = f'{self.data_subdir}/{name}.npy'
            else:
                path = f'{self.eval_data_subdir}/{name}.npy'
            
            np.save(path, np.array(self.Validation_predictions))
            self.Validation_predictions.clear() #reset 

        

    def save_errors(self, train_loss,val_loss,val_AUROC,num_batches):
        np.save(self.data_subdir + "/train_loss.npy", np.array(train_loss))
        np.save(self.data_subdir + "/val_loss.npy", np.array(val_loss))
        np.save(self.data_subdir + "/val_AUROC.npy", np.array(val_AUROC))
        
        train_loss_compressed=[]
        val_loss_stretched=[]
        
        for i in val_loss: 
            val_loss_stretched.extend([i]*num_batches)
        
        for i in range(len(val_loss)): #epochs
            start = i*num_batches
            end   = min(len(train_loss),start+num_batches)
            train_loss_compressed.append(np.mean(train_loss[start:end]))
        plt.figure(0)
        plt.plot(train_loss_compressed, color="blue", label="train loss")
        plt.plot(val_loss, color="orange", label="val loss")
        plt.legend()
        plt.savefig(self.data_subdir + "/plotLoss.png")

        plt.figure(1)
        plt.plot(val_AUROC, color="green", label="AUROC")
        plt.legend()
        plt.savefig(self.data_subdir + "/plotAUROC.png")

    def log_images(self, images, epoch, n_batch, num_batches, i_format='NCHW', normalize=True):
        """ input images are expected in format (NCHW) """
        if type(images) == np.ndarray:
            images = torch.from_numpy(images)

        if i_format == 'NHWC':
            images = images.transpose(1, 3)

        step = Logger._step(epoch, n_batch, num_batches)
        img_name = '{}/images{}'.format(self.data_subdir, '')

        # Make horizontal grid from image tensor
        horizontal_grid = vutils.make_grid(
            images, normalize=normalize, scale_each=True)

        # Add horizontal images to tensorboard
        self.writer.add_image(img_name, horizontal_grid, step)

        # Save plots
        self.save_torch_images(horizontal_grid, epoch, n_batch)

    def save_torch_images(self, horizontal_grid, epoch, n_batch):
        # Plot and save horizontal
        fig = plt.figure(figsize=(16, 16))
        plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
        plt.axis('off')
        fig.savefig('{}/epoch_{}_batch_{}.png'.format(self.data_subdir, epoch, n_batch))
        plt.close()

    @staticmethod
    def display_status(epoch, num_epochs, n_batch, num_batches, encoder_error, encoder_acc):

        # var_class = torch.autograd.variable.Variable
        
        if isinstance(encoder_error, torch.autograd.Variable):
            encoder_error = encoder_error.data.cpu().numpy()
        
        if isinstance(encoder_acc, torch.autograd.Variable):
            encoder_acc = encoder_acc.data.cpu().numpy()
        
        #print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
        #    epoch, num_epochs, n_batch, num_batches)
        #)
        #print('encoder Loss: {:.4f}, encoder Acc: {:.4f}'.format(encoder_error,encoder_acc))
        print('encoder Loss: {:.4f}'.format(encoder_error))

        # print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(d_pred_real.mean(), d_pred_fake.mean()))

    @staticmethod
    def display_Valdiation_status(epoch, num_epochs, val_loss, val_acc,val_AUROC):

        # var_class = torch.autograd.variable.Variable
        
        if isinstance(val_loss, torch.autograd.Variable):
            val_loss = val_loss.data.cpu().numpy()
        
        if isinstance(val_loss, torch.autograd.Variable):
            val_loss = val_loss.data.cpu().numpy()
        
        # print('Epoch: [{}/{}]'.format(
        #     epoch, num_epochs)
        # )
        print('Epoch: [{}/{}], Val Loss: {:.4f}, Val Acc: {:.4f}, Val AUROC: {:.4f}'.format(
            epoch, num_epochs,val_loss,val_acc, val_AUROC))

    def save_models(self, generator):
        torch.save(generator.state_dict(), f'{self.data_subdir}/generator.pt')

    def save_model(self, model,name="generator"):
        torch.save(model.state_dict(), f'{self.data_subdir}/{name}.pt')

    def save_model(self, model,name,epoch,loss,AUROC,optimizer=None):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            #'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'AUROC': AUROC,
            }, f'{self.data_subdir}/{name}.pt')
    
    def load_model(self,model,name):
        # load saved weights for generator  & intialize Models
        if not self.toEvaluate:
            model_weights_path=f'{self.data_subdir}/{name}.pt'
        else:
            model_weights_path=f'{self.server_data_subdir}/{name}.pt'

        model.load_state_dict(torch.load(model_weights_path, map_location=lambda storage, loc: storage)['model_state_dict'])
        return model

    def savefig(self,fig,filename):
        if not self.toEvaluate:
            path=f'{self.data_subdir}/{filename}.png'
        else:
            path=f'{self.eval_data_subdir}/{filename}.png'
        fig.savefig(path)

    def close(self):
        self.writer.close()

    # Private Functions
    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def save_scores(self, time, accuracy,AUROC,model_classification_report):
        if not self.toEvaluate:
            path = f'{self.data_subdir}/results.txt'
        else:
            path = f'{self.eval_data_subdir}/results.txt'

        with open(path, 'w') as file:
            file.write(f'time taken: {round(time, 4)}\n')
            file.write(f'accuracy: {round(accuracy, 4)}\n')
            file.write(f'AUROC: {round(AUROC, 4)}\n')
            file.write(model_classification_report)
