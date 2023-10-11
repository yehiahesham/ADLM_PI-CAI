from torch import Generator
from torch.utils.data import DataLoader, sampler,random_split, Subset
from sklearn.model_selection import train_test_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
import sys
sys.path.append('..')
from data_scripts.PI_CAI_Detection import PI_CAI_Detection
from models.augmentations import augmentation_contrastive, augmentation_supervised

download_path = "../data/"

def get_loader(batchSize=30, toShuffle = False, percentage=1,
               dataset="picai", modalities=["t2w"],
               multiModal=False,useAI_Segmentation=False,biasFeildCorrection=False,
               resample = False,
               targetVoxelShape= (1.0, 1.0, 3.0),targetimageSize = [220, 220, 45],
               t2w_features=[], adc_features=[], selected_features=[],
               augmentation=None,
               labels_path='/data1/practical-sose23/morphometric/picai_labels_cropped',
               image_path ='/data1/practical-sose23/morphometric/data_cropped',
               corruption=0,
               seed=42,seed2=52
               ):
    test_ratio = 0.1
    val_ratio  = 0.2
    WeightedSamplerGen = Generator().manual_seed(seed2)

    num_workers = 20
    
    if   dataset == "picai":
        
        Data = PI_CAI_Detection(image_path=image_path,labels_path=labels_path,modalities=modalities,
                                   multiModal=multiModal,useAI_Segmentation=useAI_Segmentation,biasFeildCorrection=biasFeildCorrection,transform=augmentation,
                                   targetVoxelShape=targetVoxelShape,targetimageSize=targetimageSize, resample=resample,selected_features=selected_features,corruption=corruption
                                )
        train_Lables = Data.get_Lables()
        X_train_idx, X_test_idx, y_train, y_test = train_test_split(np.arange(len(Data)), train_Lables,
                                                    test_size=test_ratio, stratify=train_Lables, random_state=seed)
        X_train_idx, X_val_idx, y_train, y_val   = train_test_split(X_train_idx, y_train,
                                                    test_size=val_ratio, stratify=y_train, random_state=seed)
        
        train_loader = DataLoader(Subset(Data,X_train_idx),batch_size=batchSize,shuffle=toShuffle, num_workers=num_workers, drop_last=True)
        val_loader   = DataLoader(Subset(Data,X_val_idx)  ,batch_size=batchSize,shuffle=toShuffle, num_workers=num_workers)
        test_loader  = DataLoader(Subset(Data,X_test_idx) ,batch_size=batchSize,shuffle=toShuffle, num_workers=num_workers)
        return train_loader,val_loader,test_loader,(y_train,y_val,y_test)
    
    elif dataset == "Balanced-picai":
        Data = PI_CAI_Detection(image_path=image_path,labels_path=labels_path,modalities=modalities,
                                   multiModal=multiModal,useAI_Segmentation=useAI_Segmentation,biasFeildCorrection=biasFeildCorrection,transform=augmentation,
                                   targetVoxelShape=targetVoxelShape,targetimageSize=targetimageSize, resample=resample,selected_features=[],corruption=corruption
                                   )
        train_Lables = Data.get_Lables()
        X_train_idx, X_test_idx, y_train, y_test = train_test_split(np.arange(len(Data)), train_Lables,
                                                    test_size=test_ratio, stratify=train_Lables, random_state=seed)
        X_train_idx, X_val_idx, y_train, y_val   = train_test_split(X_train_idx, y_train,
                                                    test_size=val_ratio, stratify=y_train, random_state=seed)
        
        # Code ref: https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264/2
        # Compute samples weight (each sample should get its own weight)
        class_sample_count = np.unique(y_train, return_counts=True)[1]
        weight = 1. / class_sample_count
        samples_weight = weight[y_train]
        WeightedRandomSampler = sampler.WeightedRandomSampler(samples_weight, len(samples_weight),generator=WeightedSamplerGen)

        train_loader = DataLoader(Subset(Data,X_train_idx),batch_size=batchSize,shuffle=toShuffle, num_workers=num_workers,sampler=WeightedRandomSampler)
        val_loader   = DataLoader(Subset(Data,X_val_idx)  ,batch_size=batchSize,shuffle=toShuffle, num_workers=num_workers)
        test_loader  = DataLoader(Subset(Data,X_test_idx) ,batch_size=batchSize,shuffle=toShuffle, num_workers=num_workers)
        return train_loader,val_loader,test_loader,(y_train,y_val,y_test)
    elif dataset == "cifar-10":
        transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root=download_path, train=True, download=True, transform=transform)
        valset = torchvision.datasets.CIFAR10(root=download_path, train=False, download=True, transform=transform)
        train_loader = DataLoader(trainset, batch_size=batchSize, shuffle=True)
        val_loader   = DataLoader(valset, batch_size=batchSize, shuffle=True )
        return train_loader,None,None,(None,None,None)
    elif dataset == "cifar-100":
        transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR100(root=download_path, train=True, download=True, transform=transform)
        valset   = torchvision.datasets.CIFAR100(root=download_path, train=False, download=True, transform=transform)
        train_loader = DataLoader(trainset, batch_size=batchSize, shuffle=True)
        val_loader   = DataLoader(valset, batch_size=batchSize, shuffle=True )
        return train_loader,None,None,(None,None,None)
    else:
        raise Exception("dataset name not correct (or not implemented)")
    
