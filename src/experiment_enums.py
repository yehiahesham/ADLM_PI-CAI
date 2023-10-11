from enum import Enum
from models.encoders import *
from models.augmentations import *
from monai.networks.nets import resnet10, resnet18, resnet50
from clip_loss import CLIPLoss

from torch import nn, optim
from lightly.loss import NTXentLoss
from experiment import Experiment
import json

PICAI_Tabular_Features = None
with open('misc_files/PICAI_Tabular_Features.json') as json_file:
    PICAI_Tabular_Features = json.load(json_file)

class ExperimentEnums(Enum):
    
#     picai_UniModal_Supervised_ResNet10 = {
#         "dataset": 'picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
#         "toShuffle": False,
#         "toTrain":True,
#         "isMultiModal":False,
#         "isContrastive": False,
#         "useAI_Segmentation":False,
#         "biasFeildCorrection":False,
#         "resample": False,
#         "modalities":["t2w"],
#         "selected_features":["patient_age","psa","psad","prostate_volume"], #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
#                               #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
#         "tabular_encoder": Tabular_Encoder,
#         "targetVoxelShape": (1,1,3),
#         "targetimageSize":[100,100,25], #[220, 220, 45],
#         "batchSize": 25,#15
#         "batchAccumulator":  1,#15
#         "percentage": 1,
#         'encoder_emb_sz':100,
#         'augmentation': augmentation_supervised,
#         'table_corruption': 0.3,
#         'encoder':EncoderNet_MONAI_resnet10,
#         "encoder_weights_path":None,
#         "encoder_optim": optim.Adam,
#         "encoder_lr": 0.0002,
#         "embClassifer": None,
#         "loss": nn.BCELoss(),
#         "epochs": 100
#     }

#     picai_UniModal_Supervised_ResNet10_WholeData = {
#         "dataset": 'picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
#         "toShuffle": False,
#         "toTrain":True,
#         "isMultiModal":False,
#         "isContrastive": False,
#         "useAI_Segmentation":True,
#         "biasFeildCorrection":False,
#         "resample": False,
#         "modalities":["t2w"],
#         "selected_features":["patient_age","psa","psad","prostate_volume"], #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
#                               #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
#         "tabular_encoder": Tabular_Encoder,
#         "targetVoxelShape": (1,1,3),
#         "targetimageSize":[100,100,25], #[220, 220, 45],
#         "batchSize": 25,#15
#         "batchAccumulator":  1,#15
#         "percentage": 1,
#         'encoder_emb_sz':100,
#         'augmentation': augmentation_supervised,
#         'table_corruption': 0.3,
#         'encoder':EncoderNet_MONAI_resnet10,
#         'encoder_weights_path':None,
#         "encoder_optim": optim.Adam,
#         "encoder_lr": 0.0002,
#         "embClassifer": None,
#         "loss": nn.BCELoss(),
#         "epochs": 100
#     }

#     Balanced_picai_UniModal_Supervised_ResNet10_WholeData = {
#         "dataset": 'Balanced-picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
#         "toShuffle": False,
#         "toTrain":True,
#         "isMultiModal":False,
#         "isContrastive": False,
#         "useAI_Segmentation":True,
#         "biasFeildCorrection":False,
#         "resample": False,
#         "modalities":["t2w"],
#         "selected_features":["patient_age","psa","psad","prostate_volume"], #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
#                               #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
#         "tabular_encoder": Tabular_Encoder,
#         "targetVoxelShape": (1,1,3),
#         "targetimageSize":[100,100,25], #[220, 220, 45],
#         "batchSize": 25,#15
#         "batchAccumulator":  1,#15
#         "percentage": 1,
#         'encoder_emb_sz':100,
#         'augmentation': augmentation_supervised,
#         'table_corruption': 0.3,
#         'encoder':EncoderNet_MONAI_resnet10,
#         'encoder_weights_path':None,
#         "encoder_optim": optim.Adam,
#         "encoder_lr": 0.0002,
#         "embClassifer": None,
#         "loss": nn.BCELoss(),
#         "epochs": 100
#     }

    
#     picai_UniModal_Supervised = {
#         "dataset": 'picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
#         "toShuffle": False,
#         "isMultiModal":False,
#         "isContrastive": False,
#         "useAI_Segmentation":True,
#         "biasFeildCorrection":False,
#         "modalities":["t2w"],
#         "selected_features":["patient_age","psa","psad","prostate_volume"],
#                          #["patient_age","psa","psad","prostate_volume","prostate_volume_T2W","original_shape_Elongation_T2W","original_glrlm_RunLengthNonUniformity_T2W","original_shape_MinorAxisLength_T2W",
#                          # "original_shape_Sphericity_T2W", # "original_firstorder_Kurtosis_T2W","original_shape_Maximum2DDiameterRow_T2W","original_glcm_Idn_T2W",
#                          # "original_shape_SurfaceVolumeRatio_T2W","original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W,
#                          # "original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"]
#         "tabular_encoder": Tabular_Encoder,
#         "targetVoxelShape": (2.0, 2.0, 6.0),
#         "targetimageSize":[195//2, 195//2, 99/6],
#         "batchSize": 64,#15
#         "batchAccumulator": 2,#15
#         "percentage": 1,
#         'encoder_emb_sz':100,
#         'table_corruption': 0.3,
#         'encoder':EncoderNet_MONAI_resnet10,
#         "encoder_optim": optim.Adam,
#         "encoder_lr": 0.0002,
#         "embClassifer": None,
#         "loss": nn.BCELoss(),
#         "epochs": 30
#     }

    # picai_UniModal_Supervised_ResNet18 = {
    #     "dataset": 'picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
    #     "toShuffle": False,
    #     "toTrain":True,
    #     "isMultiModal":False,
    #     "isContrastive": False,
    #     "useAI_Segmentation":False,
    #     "biasFeildCorrection":False,
    #     "resample": False,
    #     "modalities":["t2w"]
    #     "selected_features":["patient_age","psa","psad","prostate_volume"], #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
    #                           #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
    #     "tabular_encoder": Tabular_Encoder,
    #     "targetVoxelShape": (1,1,3),
    #     "targetimageSize":[100,100,25], #[220, 220, 45],
    #     "batchSize": 25,#15
    #     "batchAccumulator":  1,#15
    #     "percentage": 1,
    #     'encoder_emb_sz':100,
    #     'augmentation': augmentation_supervised,
    #     'table_corruption': 0.3,
    #     'encoder':EncoderNet_MONAI_resnet18,
    #     "encoder_optim": optim.Adam,
    #     "encoder_lr": 0.0002,
    #     "embClassifer": None,
    #     "loss": nn.BCELoss(),
    #     "epochs": 100
    # }
    
    # picai_UniModal_Supervised_ResNet18_WholeData = {
    #     "dataset": 'picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
    #     "toShuffle": False,
    #     "toTrain":True,
    #     "isMultiModal":False,
    #     "isContrastive": False,
    #     "useAI_Segmentation":True,
    #     "biasFeildCorrection":False,
    #     "resample": False,
    #     "modalities":["t2w"]
    #     "selected_features":["patient_age","psa","psad","prostate_volume"], #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
    #                           #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
    #     "tabular_encoder": Tabular_Encoder,
    #     "targetVoxelShape": (1,1,3),
    #     "targetimageSize":[100,100,25], #[220, 220, 45],
    #     "batchSize": 25,#15
    #     "batchAccumulator":  1,#15
    #     "percentage": 1,
    #     'encoder_emb_sz':100,
    #     'augmentation': augmentation_supervised,
    #     'table_corruption': 0.3,
    #     'encoder':EncoderNet_MONAI_resnet18,
    #     "encoder_optim": optim.Adam,
    #     "encoder_lr": 0.0002,
    #     "embClassifer": None,
    #     "loss": nn.BCELoss(),
    #     "epochs": 100
    # }

    # Balanced_picai_UniModal_Supervised_ResNet18_WholeData = {
    #     "dataset": 'Balanced-picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
    #     "toShuffle": False,
    #     "toTrain":True,
    #     "isMultiModal":False,
    #     "isContrastive": False,
    #     "useAI_Segmentation":True,
    #     "biasFeildCorrection":False,
    #     "resample": False,
    #     "modalities":["t2w"]
    #     "selected_features":["patient_age","psa","psad","prostate_volume"], #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
    #                           #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
    #     "tabular_encoder": Tabular_Encoder,
    #     "targetVoxelShape": (1,1,3),
    #     "targetimageSize":[100,100,25], #[220, 220, 45],
    #     "batchSize": 25,#15
    #     "batchAccumulator":  1,#15
    #     "percentage": 1,
    #     'encoder_emb_sz':100,
    #     'augmentation': augmentation_supervised,
    #     'table_corruption': 0.3,
    #     'encoder':EncoderNet_MONAI_resnet18,
    #     "encoder_optim": optim.Adam,
    #     "encoder_lr": 0.0002,
    #     "embClassifer": None,
    #     "loss": nn.BCELoss(),
    #     "epochs": 100
    # }
    
    # picai_UniModal_Supervised_Resnet50 = {
    #     "dataset": 'picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
    #     "toShuffle": False,
    #     "toTrain":True,
    #     "isMultiModal":False,
    #     "isContrastive": False,
    #     "useAI_Segmentation":False,
    #     "biasFeildCorrection":False,
    #     "resample": False,
    #     "modalities":["t2w"]
    #     "selected_features":["patient_age","psa","psad","prostate_volume"], #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
    #                           #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
    #     "tabular_encoder": Tabular_Encoder,
    #     "targetVoxelShape": (1,1,3),
    #     "targetimageSize":[100,100,25], #[220, 220, 45],
    #     "batchSize": 25,#15
    #     "batchAccumulator":  1,#15
    #     "percentage": 1,
    #     'encoder_emb_sz':100,
    #     'augmentation': augmentation_supervised,
    #     'table_corruption': 0.3,
    #     'encoder':EncoderNet_MONAI_resnet50,
    #     "encoder_optim": optim.Adam,
    #     "encoder_lr": 0.0002,
    #     "embClassifer": None,
    #     "loss": nn.BCELoss(),
    #     "epochs": 100
    # }

#     Balanced_picai_UniModal_Supervised_ResNet10 = {
#         "dataset": 'Balanced-picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
#         "toShuffle": False,
#         "toTrain":True,
#         "isMultiModal":False,
#         "isContrastive": False,
#         "useAI_Segmentation":False,
#         "biasFeildCorrection":False,
#         "resample": False,
#         "modalities":["t2w"],
#         "selected_features":["patient_age","psa","psad","prostate_volume"], #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
#                               #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
#         "tabular_encoder": Tabular_Encoder,
#         "targetVoxelShape": (1,1,3),
#         "targetimageSize":[100,100,25], #[220, 220, 45],
#         "batchSize": 25 ,#15
#         "batchAccumulator":  1,#15
#         "percentage": 1,
#         'encoder_emb_sz':100,
#         'augmentation': augmentation_supervised,
#         'table_corruption': 0.3,
#         'encoder':EncoderNet_MONAI_resnet10,
#         'encoder_weights_path':None,
#         "encoder_optim": optim.Adam,
#         "encoder_lr": 0.0002,
#         "embClassifer": None,
#         "loss": nn.BCELoss(),
#         "epochs": 100
#     }

    # Balanced_picai_UniModal_Supervised_ResNet18 = {
    #     "dataset": 'Balanced-picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
    #     "toShuffle": False,
    #     "toTrain":True,
    #     "isMultiModal":False,
    #     "isContrastive": False,
    #     "useAI_Segmentation":False,
    #     "biasFeildCorrection":False,
    #     "resample": False,
    #     "modalities":["t2w"]
    #     "selected_features":["patient_age","psa","psad","prostate_volume"], #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
    #                           #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
    #     "tabular_encoder": Tabular_Encoder,
    #     "targetVoxelShape": (1,1,3),
    #     "targetimageSize":[100,100,25], #[220, 220, 45],
    #     "batchSize": 10 ,#15
    #     "batchAccumulator":  1,#15
    #     "percentage": 1,
    #     'encoder_emb_sz':100,
    #     'augmentation': augmentation_supervised,
    #     'table_corruption': 0.3,
    #     'encoder':EncoderNet_MONAI_resnet18,
    #     "encoder_optim": optim.Adam,
    #     "encoder_lr": 0.0002,
    #     "embClassifer": None,
    #     "loss": nn.BCELoss(),
    #     "epochs": 100
    # }

    # Balanced_picai_UniModal_Supervised_ResNet50 = {
    #     "dataset": 'Balanced-picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
    #     "toShuffle": False,
    #     "toTrain":True,
    #     "isMultiModal":False,
    #     "isContrastive": False,
    #     "useAI_Segmentation":False,
    #     "biasFeildCorrection":False,
    #     "resample": False,
    #     "modalities":["t2w"]
    #     "selected_features":["patient_age","psa","psad","prostate_volume"], #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
    #                           #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
    #     "tabular_encoder": Tabular_Encoder,
    #     "targetVoxelShape": (1,1,3),
    #     "targetimageSize":[100,100,25], #[220, 220, 45],
    #     "batchSize": 20 ,#15
    #     "batchAccumulator":  1,#15
    #     "percentage": 1,
    #     'encoder_emb_sz':100,
    #     'augmentation': augmentation_supervised,
    #     'table_corruption': 0.3,
    #     'encoder':EncoderNet_MONAI_resnet50,
    #     "encoder_optim": optim.Adam,
    #     "encoder_lr": 0.0002,
    #     "embClassifer": None,
    #     "loss": nn.BCELoss(),
    #     "epochs": 100
    # }

    # picai_Unimodal_Contrastive_ResNet10 = {
    #     "dataset": 'picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
    #     "toShuffle": False,
    #     "toTrain":True,
    #     "isMultiModal":False,
    #     "isContrastive": True,
    #     "useAI_Segmentation":False,
    #     "biasFeildCorrection":False,
    #     "resample": False,
    #     "modalities":["t2w"]
    #     "selected_features":["patient_age","psa","psad","prostate_volume"], #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
    #                           #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
    #     "tabular_encoder": Tabular_Encoder,
    #     "targetVoxelShape": (1.0, 1.0, 3.0),
    #     "targetimageSize":[50, 50, 45], #[220, 220, 45],
    #     "batchSize": 5,
    #     "batchAccumulator":  1,#15
    #     "percentage": 1,
    #     'encoder_emb_sz':100,
    #     'augmentation': augmentation_contrastive,
    #     'table_corruption': 0.3,
    #     'encoder':Contrastive_Encoder_resnet10,
    #     "encoder_optim": optim.Adam,
    #     "encoder_lr": 0.0002,
    #     "embClassifer": None,
    #     "loss": NTXentLoss(),
    #     "epochs": 3
    # }

    # picai_Unimodal_Contrastive_ResNet10 = {
    #     "dataset": 'picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
    #     "toShuffle": False,
    #     "toTrain":True,
    #     "isMultiModal":False,
    #     "isContrastive": True,
    #     "useAI_Segmentation":True,
    #     "biasFeildCorrection":False,
    #     "resample": False,
    #     "modalities":["t2w"]
    #     "selected_features":["patient_age","psa","psad","prostate_volume"], #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
    #                           #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
    #     "tabular_encoder": Tabular_Encoder,
    #     "targetVoxelShape": (2.0, 2.0, 6.0),
    #     "targetimageSize":[195//2, 195//2, 99/6], #[220, 220, 45],
    #     "batchSize": 64,
    #     "batchAccumulator":  1,#15
    #     "percentage": 1,
    #     'encoder_emb_sz':100,
    #     'augmentation': augmentation_contrastive,
    #     'table_corruption': 0.3,
    #     'encoder':Contrastive_Encoder_resnet10,
    #     "encoder_optim": optim.Adam,
    #     "encoder_lr": 0.0002,
    #     "embClassifer": None,
    #     "loss": NTXentLoss(),
    #     "epochs": 20
    # }

    # Balanced_picai_UniModal_Supervised_pretrained_ResNet18 = {
    #     "dataset": 'Balanced-picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
    #     "isContrastive": False,
    #     "toShuffle": False,
    #     "useAI_Segmentation":False,
    #     "toTrain":True,
    #     "biasFeildCorrection":False,
    #     "isMultiModal":False,
    #     "selected_features":["patient_age","psa","psad","prostate_volume"], 
    #                       #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
    #                       # "original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
    #     "resample": False,
    #     "modalities":["t2w"],
    #     "targetVoxelShape": (1.0, 1.0, 3.0),
    #     "targetimageSize":[100, 100, 25],
    #     "batchSize": 50,#15
    #     "batchAccumulator": 1,#15
    #     "percentage": 1,
    #     'encoder_emb_sz':100,
    #     'augmentation': augmentation_supervised,
    #     'table_corruption': 0.3,
    #     'encoder':EncoderNet_MONAI_resnet18_pretrained,
    #     "encoder_weights_path":None,
    #     "tabular_encoder": Tabular_Encoder,
    #     "encoder_optim": optim.Adam,
    #     "encoder_lr": 0.0001,
    #     "embClassifer": None,
    #     "loss": nn.BCELoss(),
    #     "epochs": 100
    # }

    # Balanced_picai_UniModal_Supervised_pretrained_ResNet18_frozen = {
    #     "dataset": 'Balanced-picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
    #     "isContrastive": False,
    #     "toShuffle": False,
    #     "useAI_Segmentation":False,
    #     "isTesting":False,
    #     "biasFeildCorrection":False,
    #     "isMultiModal":False,
    #     "selected_features":["patient_age","psa","psad","prostate_volume"], #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
    #                           #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
    #     "resample": False,
    #     "modalities":["t2w"],
    #     "targetVoxelShape": (1.0, 1.0, 3.0),
    #     "targetimageSize":[100, 100, 25],
    #     "batchSize": 50,#15
    #     "batchAccumulator": 1,#15
    #     "percentage": 1,
    #     'encoder_emb_sz':100,
    #     'augmentation': augmentation_supervised,
    #     'table_corruption': 0.3,
    #     'encoder':EncoderNet_MONAI_resnet18_pretrained_frozen,
    #     'encoder_weights_path': None,
    #     "tabular_encoder": Tabular_Encoder,
    #     "encoder_optim": optim.Adam,
    #     "encoder_lr": 0.001,
    #     "embClassifer": None,
    #     "loss": nn.BCELoss(),
    #     "epochs": 100
    # }
    '''
    Balanced_picai_UniModal_Res10_300epochs_finetuned_larger_batches = {
        "dataset": 'Balanced-picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
        "isContrastive": False,
        "toShuffle": False,
        "toTrain":True,
        "useAI_Segmentation":False,
        "isTesting":False,
        "biasFeildCorrection":False,
        "isMultiModal":False,
        "t2w_features":[],#["original_gldm_LowGrayLevelEmphasis","original_gldm_SmallDependenceEmphasis"],
        "adc_features":[],#["original_gldm_LowGrayLevelEmphasis","original_gldm_SmallDependenceEmphasis"],
        "resample": True,
        "modalities":["t2w"],
        "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
        "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
        "batchSize": 100,#15
        "batchAccumulator": 1,#15
        "percentage": 1,
        'encoder_emb_sz':100,
        'augmentation': augmentation_supervised,
        'table_corruption': 0.3,
        'encoder': Contrastive_Encoder_resnet10,
        'encoder_weights_path': "/data1/practical-sose23/morphometric/results_arno/picai/picai_Unimodal_Contrastive_Res10_300epochs/encoder.pt",
        "tabular_encoder": Tabular_Encoder,
        "encoder_optim": optim.Adam,
        "encoder_lr": 0.00005,
        "embClassifer": None,
        "loss": nn.BCELoss(),
        "epochs": 50
    }'''

    # Balanced_picai_UniModal_test_loading_pretrained_weights = {
    #     "dataset": 'Balanced-picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
    #     "isContrastive": False,
    #     "toShuffle": False,
    #     "toTrain":True,
    #     "useAI_Segmentation":False,
    #     "isTesting":False,
    #     "biasFeildCorrection":False,
    #     "isMultiModal":False,
    #     "selected_features":["patient_age","psa","psad","prostate_volume"], #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
    #                           #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
    #     "resample": False,
    #     "modalities":["t2w"],
    #     "targetVoxelShape": (1.0, 1.0, 3.0),
    #     "targetimageSize":[100, 100, 25],
    #     "batchSize": 50,#15
    #     "batchAccumulator": 1,#15
    #     "percentage": 1,
    #     'encoder_emb_sz':100,
    #     'augmentation': augmentation_supervised,
    #     'table_corruption': 0.3,
    #     'encoder': Contrastive_Encoder_resnet10,
    #     'encoder_weights_path': "/data1/practical-sose23/morphometric/results_arno/picai/picai_Unimodal_Contrastive/encoder.pt",
    #     "tabular_encoder": Tabular_Encoder,
    #     "encoder_optim": optim.Adam,
    #     "encoder_lr": 0.001,
    #     "embClassifer": None,
    #     "loss": nn.BCELoss(),
    #     "epochs": 1
    # }

    # Balanced_picai_UniModal_Supervised_pretrained_ResNet10_frozen = {
    #     "dataset": 'Balanced-picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
    #     "isContrastive": False,
    #     "toShuffle": False,
    #     "toTrain":True,
    #     "useAI_Segmentation":False,
    #     "isTesting":False,
    #     "biasFeildCorrection":False,
    #     "isMultiModal":False,
    #     "selected_features":["patient_age","psa","psad","prostate_volume"], #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
    #                           #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
    #     "resample": False,
    #     "modalities":["t2w"],
    #     "targetVoxelShape": (1.0, 1.0, 3.0),
    #     "targetimageSize":[100, 100, 25],
    #     "batchSize": 50,#15
    #     "batchAccumulator": 1,#15
    #     "percentage": 1,
    #     'encoder_emb_sz':100,
    #     'augmentation': augmentation_supervised,
    #     'table_corruption': 0.3,
    #     'encoder': Contrastive_Encoder_resnet10,
    #     'encoder_weights_path': "/data1/practical-sose23/morphometric/results_arno/picai/picai_Unimodal_Contrastive/encoder.pt",
    #     "tabular_encoder": Tabular_Encoder,
    #     "encoder_optim": optim.Adam,
    #     "encoder_lr": 0.001,
    #     "embClassifer": None,
    #     "loss": nn.BCELoss(),
    #     "epochs": 1
    # }

    # Balanced_picai_UniModal_Supervised_pretrained_ResNet10_frozen = {
    #     "dataset": 'Balanced-picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
    #     "isContrastive": False,
    #     "toShuffle": False,
    #     "useAI_Segmentation":False,
    #     "toTrain":True,
    #     "biasFeildCorrection":False,
    #     "isMultiModal":False,
    #     "selected_features":["patient_age","psa","psad","prostate_volume"], #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
    #                           #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
    #     "resample": False,
    #     "modalities":["t2w"],
    #     "targetVoxelShape": (1.0, 1.0, 3.0),
    #     "targetimageSize":[100, 100, 25],
    #     "batchSize": 10,#15
    #     "batchAccumulator": 3,#15
    #     "percentage": 1,
    #     'encoder_emb_sz':100,
    #     'augmentation': augmentation_supervised,
    #     'table_corruption': 0.3,
    #     'encoder':EncoderNet_MONAI_resnet10_pretrained,
    #     "tabular_encoder": Tabular_Encoder,
    #     "encoder_optim": optim.Adam,
    #     "encoder_lr": 0.001,
    #     "embClassifer": None,
    #     "loss": nn.BCELoss(),
    #     "epochs": 100
    # }
    # 

    """
    picai_Unimodal_Contrastive_Res10_300epochs = {

    # picai_Unimodal_Contrastive = {
    #      "dataset": 'picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
    #      "toShuffle": False,
    #      "toTrain":True,
    #      "isMultiModal":False,
    #      "selected_features":["patient_age","psa","psad","prostate_volume"],
    #                       #["patient_age","psa","psad","prostate_volume","prostate_volume_T2W","original_shape_Elongation_T2W","original_glrlm_RunLengthNonUniformity_T2W","original_shape_MinorAxisLength_T2W",
    #                       # "original_shape_Sphericity_T2W", # "original_firstorder_Kurtosis_T2W","original_shape_Maximum2DDiameterRow_T2W","original_glcm_Idn_T2W",
    #                       # "original_shape_SurfaceVolumeRatio_T2W","original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W,
    #                       # "original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"]    
    #      "isContrastive": True,
    #      "useAI_Segmentation":True,
    #      "biasFeildCorrection":False,
    #      "resample": True,
    #      "modalities":["t2w"],
    #      "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
    #      "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
    #      "batchSize": 100,
    #      "batchAccumulator": 2,#15
    #      "percentage": 1,
    #      'encoder_emb_sz':100,
    #      'augmentation':augmentation_contrastive,
    #      'table_corruption': 0.3,
    #      'encoder_weights_path':None,
    #      'encoder':Contrastive_Encoder_resnet10,
    #      "tabular_encoder": Tabular_Encoder,
    #      "encoder_optim": optim.Adam,
    #      "encoder_lr": 0.002,
    #      "embClassifer": None,
    #      "loss": NTXentLoss(),#CLIPLoss(0.1,0.5),#NTXentLoss(),
    #      "epochs": 100
    #  }

    # picai_Unimodal_Contrastive_ResNet10_Aug_0_2 = {
    #      "dataset": 'picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
    #      "toShuffle": False,
    #      "toTrain":True,
    #      "isMultiModal":False,
    #      "selected_features":["patient_age","psa","psad","prostate_volume"],
    #                       #["patient_age","psa","psad","prostate_volume","prostate_volume_T2W","original_shape_Elongation_T2W","original_glrlm_RunLengthNonUniformity_T2W","original_shape_MinorAxisLength_T2W",
    #                       # "original_shape_Sphericity_T2W", # "original_firstorder_Kurtosis_T2W","original_shape_Maximum2DDiameterRow_T2W","original_glcm_Idn_T2W",
    #                       # "original_shape_SurfaceVolumeRatio_T2W","original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W,
    #                       # "original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"]
    #      "isContrastive": True,
    #      "useAI_Segmentation":True,
    #      "biasFeildCorrection":False,
    #      "resample": True,
    #      "modalities":["t2w"],
    #      "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
    #      "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
    #      "batchSize": 100,
    #      "batchAccumulator": 2,#15
    #      "percentage": 1,
    #      'encoder_emb_sz':100,
    #      'augmentation':augmentation_contrastive,
    #      'table_corruption': 0.3,
    #      'encoder_weights_path':None,
    #      'encoder':Contrastive_Encoder_resnet10,
    #      "tabular_encoder": Tabular_Encoder,
    #      "encoder_optim": optim.Adam,
    #      "encoder_lr": 0.002,
    #      "embClassifer": None,
    #      "loss": NTXentLoss(),#CLIPLoss(0.1,0.5),#NTXentLoss(),
    #      "epochs": 100
    #  }

    # picai_Unimodal_Contrastive_TrainClsHead = {
    #      "dataset": 'Balanced-picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
    #      "toShuffle": False,
    #      "toTrain":True,
    #      "isMultiModal":False,
    #      "selected_features":["patient_age","psa","psad","prostate_volume"], 
    #                         #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
    #                         #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
    #      "isContrastive": False,
    #      "useAI_Segmentation":True,
    #      "biasFeildCorrection":False,
    #      "resample": True,
    #      "modalities":["t2w"],
    #      "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
    #      "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
    #      "batchSize": 125,
    #      "batchAccumulator": 1,#15
    #      "percentage": 1,
    #      'encoder_emb_sz':100,
    #      'augmentation':augmentation_supervised,
    #      'table_corruption': 0.3,
    #      'encoder_weights_path':"/u/home/ahy/adlm_ss2023_mmcl/results/picai/picai_Unimodal_Contrastive_ResNet10_Aug_0_2/encoder.pt",
    #      'encoder':Contrastive_Encoder_resnet10,
    #      "tabular_encoder": Tabular_Encoder,
    #      "encoder_optim": optim.Adam,
    #      "encoder_lr": 0.0002,
    #      "embClassifer": None,
    #      "loss": nn.BCELoss(), #[NTXentLoss(),nn.BCELoss(), #CLIPLoss(0.1,0.5),#NTXentLoss()]
    #      "epochs": 15
    #  }
    
    picai_Multmodal_Contrastive_Tabular_ResNet10  = {
         "dataset": 'picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
         "toShuffle": False,
         "toTrain":True,
         "isMultiModal":True,
         "selected_features":["patient_age","psa","psad","prostate_volume"],
         "isContrastive": True,
         "useAI_Segmentation":True,
         "biasFeildCorrection":False,
         "resample": True,
         "modalities":["t2w"],
         "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
         "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
         "batchSize": 100,
         "batchAccumulator": 2,#15
         "percentage": 1,
         'encoder_emb_sz':100,
         'augmentation':augmentation_contrastive,
         'table_corruption': 0.3,
         'encoder_weights_path':None,
         'encoder':Contrastive_Encoder_resnet10,
         "tabular_encoder": Tabular_Encoder,
         "encoder_optim": optim.Adam,
         "encoder_lr": 0.0002,
         "embClassifer": None,
         "loss": NTXentLoss(),#CLIPLoss(0.1,0.5),#NTXentLoss(),
         "epochs": 300
     }"""
    
    
    """
    picai_Multimodal_Contrastive_Res10_300epochs = {
         "dataset": 'picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
         "toShuffle": False,
         "toTrain":True,
         "isMultiModal":True,
         "t2w_features":["prostate_volume","original_shape_Elongation_TW2","original_glrlm_RunLengthNonUniformity_TW2","original_shape_MinorAxisLength_TW2","original_shape_Sphericity_TW2","original_firstorder_Kurtosis_TW2","original_shape_Maximum2DDiameterRow_TW2","original_glcm_Idn_TW2","original_shape_SurfaceVolumeRatio_TW2"],#["original_gldm_LowGrayLevelEmphasis","original_gldm_SmallDependenceEmphasis"],
         "adc_features":[],#["original_gldm_LowGrayLevelEmphasis","original_gldm_SmallDependenceEmphasis"],
         "isContrastive": True,
         "useAI_Segmentation":True,
         "biasFeildCorrection":False,
         "resample": True,
         "modalities":["t2w"],
         "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
         "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
         "batchSize": 100,
         "batchAccumulator": 2,#15
         "percentage": 1,
         'encoder_emb_sz':100,
         'augmentation':augmentation_contrastive,
         'table_corruption': 0.3,
         'encoder_weights_path': None,
         'encoder':Contrastive_Encoder_resnet10,
         "tabular_encoder": Tabular_Encoder,
         "encoder_optim": optim.Adam,
         "encoder_lr": 0.001,
         "embClassifer": None,
         "loss": NTXentLoss(),#CLIPLoss(0.1,0.5),#NTXentLoss(),
         "epochs": 1
     }"""
    
    """
    picai_Unimodal_Contrastive_ResNet10_500e_temp01  = {
         "dataset": 'picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
         "toShuffle": False,
         "toTrain":True,
         "isMultiModal":False,
         "selected_features":["patient_age","psa","psad","prostate_volume"],
         "isContrastive": True,
         "useAI_Segmentation":True,
         "biasFeildCorrection":False,
         "resample": True,
         "modalities":["t2w"],
         "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
         "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
         "batchSize": 100,
         "batchAccumulator": 1,#15
         "percentage": 1,
         'encoder_emb_sz':100,
         'augmentation':augmentation_contrastive,
         'table_corruption': 0.3,
         'encoder_weights_path':None,
         'encoder':Contrastive_Encoder_resnet10,
         "tabular_encoder": Tabular_Encoder,
         "encoder_optim": optim.Adam,
         "encoder_lr": 0.002,
         "embClassifer": None,
         "loss": NTXentLoss(temperature=0.1),#CLIPLoss(0.1,0.5),#CLIPLoss(0.1,0.5),#NTXentLoss(),
         "epochs": 500
     }"""
    
    """ picai_Unimodal_Contrastive_ResNet10_500e  = {
         "dataset": 'picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
         "toShuffle": False,
         "toTrain":True,
         "isMultiModal":False,
         "selected_features":["patient_age","psa","psad","prostate_volume"],
         "isContrastive": True,
         "useAI_Segmentation":True,
         "biasFeildCorrection":False,
         "resample": True,
         "modalities":["t2w"],
         "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
         "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
         "batchSize": 100,
         "batchAccumulator": 1,#15
         "percentage": 1,
         'encoder_emb_sz':100,
         'augmentation':augmentation_contrastive,
         'table_corruption': 0.3,
         'encoder_weights_path':None,
         'encoder':Contrastive_Encoder_resnet10,
         "tabular_encoder": Tabular_Encoder,
         "encoder_optim": optim.Adam,
         "encoder_lr": 0.002,
         "embClassifer": None,
         "loss": NTXentLoss(),#CLIPLoss(0.1,0.5),#CLIPLoss(0.1,0.5),#NTXentLoss(),
         "epochs": 500
     } """
    
    """ picai_Unimodal_Contrastive_ResNet10_500e_CLS_head = {
         "dataset": 'Balanced-picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
         "toShuffle": False,
         "toTrain":True,
         "isMultiModal":False,
         "selected_features":["patient_age","psa","psad","prostate_volume"], 
                            #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
                            #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
         "isContrastive": False,
         "useAI_Segmentation":True,
         "biasFeildCorrection":False,
         "resample": True,
         "modalities":["t2w"],
         "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
         "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
         "batchSize": 125,
         "batchAccumulator": 1,#15
         "percentage": 1,
         'encoder_emb_sz':100,
         'augmentation':augmentation_supervised,
         'table_corruption': 0.3,
         'encoder_weights_path':"/data1/practical-sose23/morphometric/results_arno/picai/picai_Unimodal_Contrastive_ResNet10_500e/encoder.pt",
         'encoder':Contrastive_Encoder_resnet10,
         "tabular_encoder": Tabular_Encoder,
         "encoder_optim": optim.Adam,
         "encoder_lr": 0.0002,
         "embClassifer": None,
         "loss": nn.BCELoss(), #[NTXentLoss(),nn.BCELoss(), #CLIPLoss(0.1,0.5),#NTXentLoss()]
         "epochs": 50
     } """

    """ picai_Multmodal_Contrastive_Morphometric_ResNet10_500e  = {
         "dataset": 'picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
         "toShuffle": False,
         "toTrain":True,
         "isMultiModal":True,
         "selected_features":PICAI_Tabular_Features["morphometric"],
         "isContrastive": True,
         "useAI_Segmentation":True,
         "biasFeildCorrection":False,
         "resample": True,
         "modalities":["t2w"],
         "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
         "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
         "batchSize": 100,
         "batchAccumulator": 2,#15
         "percentage": 1,
         'encoder_emb_sz':100,
         'augmentation':augmentation_contrastive,
         'table_corruption': 0.3,
         'encoder_weights_path':None,
         'encoder':Contrastive_Encoder_resnet10,
         "tabular_encoder": Tabular_Encoder,
         "encoder_optim": optim.Adam,
         "encoder_lr": 0.002,
         "embClassifer": None,
         "loss": CLIPLoss(0.5,0.5),#CLIPLoss(0.1,0.5),#NTXentLoss(),
         "epochs": 500
    }   """

    """ picai_Multmodal_Contrastive_Morphometric_ResNet10_500e_CLS_head = {
         "dataset": 'Balanced-picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
         "toShuffle": False,
         "toTrain":True,
         "isMultiModal":False,
         "selected_features":["patient_age","psa","psad","prostate_volume"], 
                            #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
                            #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
         "isContrastive": False,
         "useAI_Segmentation":True,
         "biasFeildCorrection":False,
         "resample": True,
         "modalities":["t2w"],
         "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
         "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
         "batchSize": 125,
         "batchAccumulator": 1,#15
         "percentage": 1,
         'encoder_emb_sz':100,
         'augmentation':augmentation_supervised,
         'table_corruption': 0.3,
         'encoder_weights_path':"/data1/practical-sose23/morphometric/results_arno/picai/picai_Multmodal_Contrastive_Morphometric_ResNet10_500e/encoder.pt",
         'encoder':Contrastive_Encoder_resnet10,
         "tabular_encoder": Tabular_Encoder,
         "encoder_optim": optim.Adam,
         "encoder_lr": 0.0002,
         "embClassifer": None,
         "loss": nn.BCELoss(), #[NTXentLoss(),nn.BCELoss(), #CLIPLoss(0.1,0.5),#NTXentLoss()]
         "epochs": 50
     } """

    """ picai_Multmodal_Contrastive_Original_ResNet10_500e  = {
         "dataset": 'picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
         "toShuffle": False,
         "toTrain":True,
         "isMultiModal":True,
         "selected_features":PICAI_Tabular_Features["original"],
         "isContrastive": True,
         "useAI_Segmentation":True,
         "biasFeildCorrection":False,
         "resample": True,
         "modalities":["t2w"],
         "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
         "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
         "batchSize": 100,
         "batchAccumulator": 2,#15
         "percentage": 1,
         'encoder_emb_sz':100,
         'augmentation':augmentation_contrastive,
         'table_corruption': 0.3,
         'encoder_weights_path':None,
         'encoder':Contrastive_Encoder_resnet10,
         "tabular_encoder": Tabular_Encoder,
         "encoder_optim": optim.Adam,
         "encoder_lr": 0.002,
         "embClassifer": None,
         "loss": CLIPLoss(0.5,0.5),#CLIPLoss(0.1,0.5),#NTXentLoss(),
         "epochs": 500
    }  """

    """ picai_Multmodal_Contrastive_Original_ResNet10_500e_CLS_head = {
         "dataset": 'Balanced-picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
         "toShuffle": False,
         "toTrain":True,
         "isMultiModal":False,
         "selected_features":["patient_age","psa","psad","prostate_volume"], 
                            #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
                            #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
         "isContrastive": False,
         "useAI_Segmentation":True,
         "biasFeildCorrection":False,
         "resample": True,
         "modalities":["t2w"],
         "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
         "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
         "batchSize": 125,
         "batchAccumulator": 1,#15
         "percentage": 1,
         'encoder_emb_sz':100,
         'augmentation':augmentation_supervised,
         'table_corruption': 0.3,
         'encoder_weights_path':"/data1/practical-sose23/morphometric/results_arno/picai/picai_Multmodal_Contrastive_Original_ResNet10_500e/encoder.pt",
         'encoder':Contrastive_Encoder_resnet10,
         "tabular_encoder": Tabular_Encoder,
         "encoder_optim": optim.Adam,
         "encoder_lr": 0.0002,
         "embClassifer": None,
         "loss": nn.BCELoss(), #[NTXentLoss(),nn.BCELoss(), #CLIPLoss(0.1,0.5),#NTXentLoss()]
         "epochs": 50
     } """
    
#     picai_Multmodal_Contrastive_Original_ResNet10_500e_CLS_head_3l_head = {
#          "dataset": 'Balanced-picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
#          "toShuffle": False,
#          "toTrain":True,
#          "isMultiModal":False,
#          "selected_features":["patient_age","psa","psad","prostate_volume"], 
#                             #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
#                             #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
#          "isContrastive": False,
#          "useAI_Segmentation":True,
#          "biasFeildCorrection":False,
#          "resample": True,
#          "modalities":["t2w"],
#          "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
#          "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
#          "batchSize": 125,
#          "batchAccumulator": 1,#15
#          "percentage": 1,
#          'encoder_emb_sz':100,
#          'augmentation':augmentation_supervised,
#          'table_corruption': 0.3,
#          'encoder_weights_path':"/data1/practical-sose23/morphometric/results_arno/picai/picai_Multmodal_Contrastive_Original_ResNet10_500e/encoder.pt",
#          'encoder':Contrastive_Encoder_resnet10,
#          "tabular_encoder": Tabular_Encoder,
#          "encoder_optim": optim.Adam,
#          "encoder_lr": 0.0002,
#          "embClassifer": None,
#          "loss": nn.BCELoss(), #[NTXentLoss(),nn.BCELoss(), #CLIPLoss(0.1,0.5),#NTXentLoss()]
#          "epochs": 100
#      } 

    """ picai_Multmodal_Contrastive_lasso_morph_ResNet10_500e  = {
         "dataset": 'picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
         "toShuffle": False,
         "toTrain":True,
         "isMultiModal":True,
         "selected_features":PICAI_Tabular_Features["Lasso-morphometric"],
         "isContrastive": True,
         "useAI_Segmentation":True,
         "biasFeildCorrection":False,
         "resample": True,
         "modalities":["t2w"],
         "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
         "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
         "batchSize": 100,
         "batchAccumulator": 2,#15
         "percentage": 1,
         'encoder_emb_sz':100,
         'augmentation':augmentation_contrastive,
         'table_corruption': 0.3,
         'encoder_weights_path':None,
         'encoder':Contrastive_Encoder_resnet10,
         "tabular_encoder": Tabular_Encoder,
         "encoder_optim": optim.Adam,
         "encoder_lr": 0.002,
         "embClassifer": None,
         "loss": CLIPLoss(0.5,0.5),#CLIPLoss(0.1,0.5),#NTXentLoss(),
         "epochs": 500
    } """

    """ picai_Multmodal_Contrastive_lasso_morph_ResNet10_500e_CLS_head = {
         "dataset": 'Balanced-picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
         "toShuffle": False,
         "toTrain":True,
         "isMultiModal":False,
         "selected_features":["patient_age","psa","psad","prostate_volume"], 
                            #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
                            #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
         "isContrastive": False,
         "useAI_Segmentation":True,
         "biasFeildCorrection":False,
         "resample": True,
         "modalities":["t2w"],
         "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
         "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
         "batchSize": 125,
         "batchAccumulator": 1,#15
         "percentage": 1,
         'encoder_emb_sz':100,
         'augmentation':augmentation_supervised,
         'table_corruption': 0.3,
         'encoder_weights_path':"/data1/practical-sose23/morphometric/results_arno/picai/picai_Multmodal_Contrastive_lasso_morph_ResNet10_500e/encoder.pt",
         'encoder':Contrastive_Encoder_resnet10,
         "tabular_encoder": Tabular_Encoder,
         "encoder_optim": optim.Adam,
         "encoder_lr": 0.0002,
         "embClassifer": None,
         "loss": nn.BCELoss(), #[NTXentLoss(),nn.BCELoss(), #CLIPLoss(0.1,0.5),#NTXentLoss()]
         "epochs": 50
     } """
    
    """ picai_Multmodal_Contrastive_lasso_morph_ResNet10_500e_CLS_head_3l_head = {
         "dataset": 'Balanced-picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
         "toShuffle": False,
         "toTrain":True,
         "isMultiModal":False,
         "selected_features":["patient_age","psa","psad","prostate_volume"], 
                            #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
                            #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
         "isContrastive": False,
         "useAI_Segmentation":True,
         "biasFeildCorrection":False,
         "resample": True,
         "modalities":["t2w"],
         "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
         "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
         "batchSize": 125,
         "batchAccumulator": 1,#15
         "percentage": 1,
         'encoder_emb_sz':100,
         'augmentation':augmentation_supervised,
         'table_corruption': 0.3,
         'encoder_weights_path':"/data1/practical-sose23/morphometric/results_arno/picai/picai_Multmodal_Contrastive_lasso_morph_ResNet10_500e/encoder.pt",
         'encoder':Contrastive_Encoder_resnet10,
         "tabular_encoder": Tabular_Encoder,
         "encoder_optim": optim.Adam,
         "encoder_lr": 0.0002,
         "embClassifer": None,
         "loss": nn.BCELoss(), #[NTXentLoss(),nn.BCELoss(), #CLIPLoss(0.1,0.5),#NTXentLoss()]
         "epochs": 100
     } """
    
    """ picai_Multmodal_Contrastive_lasso_morph_ResNet10_500e_low_corruption  = {
         "dataset": 'picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
         "toShuffle": False,
         "toTrain":True,
         "isMultiModal":True,
         "selected_features":PICAI_Tabular_Features["Lasso-morphometric"],
         "isContrastive": True,
         "useAI_Segmentation":True,
         "biasFeildCorrection":False,
         "resample": True,
         "modalities":["t2w"],
         "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
         "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
         "batchSize": 100,
         "batchAccumulator": 2,#15
         "percentage": 1,
         'encoder_emb_sz':100,
         'augmentation':augmentation_contrastive,
         'table_corruption': 0.2,
         'encoder_weights_path':None,
         'encoder':Contrastive_Encoder_resnet10,
         "tabular_encoder": Tabular_Encoder,
         "encoder_optim": optim.Adam,
         "encoder_lr": 0.002,
         "embClassifer": None,
         "loss": CLIPLoss(0.5,0.5),#CLIPLoss(0.1,0.5),#NTXentLoss(),
         "epochs": 500
    } """

    """ picai_Multmodal_Contrastive_lasso_morph_ResNet10_500e_low_corruption_CLS_head = {
         "dataset": 'Balanced-picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
         "toShuffle": False,
         "toTrain":True,
         "isMultiModal":False,
         "selected_features":["patient_age","psa","psad","prostate_volume"], 
                            #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
                            #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
         "isContrastive": False,
         "useAI_Segmentation":True,
         "biasFeildCorrection":False,
         "resample": True,
         "modalities":["t2w"],
         "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
         "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
         "batchSize": 125,
         "batchAccumulator": 1,#15
         "percentage": 1,
         'encoder_emb_sz':100,
         'augmentation':augmentation_supervised,
         'table_corruption': 0.3,
         'encoder_weights_path':"results/picai/picai_Multmodal_Contrastive_lasso_morph_ResNet10_500e_low_corruption/encoder.pt",
         'encoder':Contrastive_Encoder_resnet10,
         "tabular_encoder": Tabular_Encoder,
         "encoder_optim": optim.Adam,
         "encoder_lr": 0.0002,
         "embClassifer": None,
         "loss": nn.BCELoss(), #[NTXentLoss(),nn.BCELoss(), #CLIPLoss(0.1,0.5),#NTXentLoss()]
         "epochs": 50
     } """

    """ picai_Multmodal_Contrastive_lasso_morph_ResNet10_500e_high_corruption  = {
         "dataset": 'picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
         "toShuffle": False,
         "toTrain":True,
         "isMultiModal":True,
         "selected_features":PICAI_Tabular_Features["Lasso-morphometric"],
         "isContrastive": True,
         "useAI_Segmentation":True,
         "biasFeildCorrection":False,
         "resample": True,
         "modalities":["t2w"],
         "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
         "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
         "batchSize": 100,
         "batchAccumulator": 2,#15
         "percentage": 1,
         'encoder_emb_sz':100,
         'augmentation':augmentation_contrastive,
         'table_corruption': 0.4,
         'encoder_weights_path':None,
         'encoder':Contrastive_Encoder_resnet10,
         "tabular_encoder": Tabular_Encoder,
         "encoder_optim": optim.Adam,
         "encoder_lr": 0.002,
         "embClassifer": None,
         "loss": CLIPLoss(0.5,0.5),#CLIPLoss(0.1,0.5),#NTXentLoss(),
         "epochs": 500
    } """


    """ picai_Multmodal_Contrastive_lasso_morph_ResNet10_500e_high_corruption_CLS_head = {
         "dataset": 'Balanced-picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
         "toShuffle": False,
         "toTrain":True,
         "isMultiModal":False,
         "selected_features":["patient_age","psa","psad","prostate_volume"], 
                            #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
                            #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
         "isContrastive": False,
         "useAI_Segmentation":True,
         "biasFeildCorrection":False,
         "resample": True,
         "modalities":["t2w"],
         "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
         "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
         "batchSize": 125,
         "batchAccumulator": 1,#15
         "percentage": 1,
         'encoder_emb_sz':100,
         'augmentation':augmentation_supervised,
         'table_corruption': 0.3,
         'encoder_weights_path':"results/picai/picai_Multmodal_Contrastive_lasso_morph_ResNet10_500e_high_corruption/encoder.pt",
         'encoder':Contrastive_Encoder_resnet10,
         "tabular_encoder": Tabular_Encoder,
         "encoder_optim": optim.Adam,
         "encoder_lr": 0.0002,
         "embClassifer": None,
         "loss": nn.BCELoss(), #[NTXentLoss(),nn.BCELoss(), #CLIPLoss(0.1,0.5),#NTXentLoss()]
         "epochs": 50
     } """

    """ picai_Multmodal_Contrastive_all_morph_ResNet10_500e  = {
         "dataset": 'picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
         "toShuffle": False,
         "toTrain":True,
         "isMultiModal":True,
         "selected_features":PICAI_Tabular_Features["morphometric"],
         "isContrastive": True,
         "useAI_Segmentation":True,
         "biasFeildCorrection":False,
         "resample": True,
         "modalities":["t2w"],
         "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
         "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
         "batchSize": 100,
         "batchAccumulator": 2,#15
         "percentage": 1,
         'encoder_emb_sz':100,
         'augmentation':augmentation_contrastive,
         'table_corruption': 0.3,
         'encoder_weights_path':None,
         'encoder':Contrastive_Encoder_resnet10,
         "tabular_encoder": Tabular_Encoder,
         "encoder_optim": optim.Adam,
         "encoder_lr": 0.002,
         "embClassifer": None,
         "loss": CLIPLoss(0.5,0.5),#CLIPLoss(0.1,0.5),#NTXentLoss(),
         "epochs": 500
    } """

    """ picai_Multmodal_Contrastive_all_morph_ResNet10_500e_CLS_head = {
         "dataset": 'Balanced-picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
         "toShuffle": False,
         "toTrain":True,
         "isMultiModal":False,
         "selected_features":["patient_age","psa","psad","prostate_volume"], 
                            #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
                            #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
         "isContrastive": False,
         "useAI_Segmentation":True,
         "biasFeildCorrection":False,
         "resample": True,
         "modalities":["t2w"],
         "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
         "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
         "batchSize": 125,
         "batchAccumulator": 1,#15
         "percentage": 1,
         'encoder_emb_sz':100,
         'augmentation':augmentation_supervised,
         'table_corruption': 0.3,
         'encoder_weights_path':"/data1/practical-sose23/morphometric/results_arno/picai/picai_Multmodal_Contrastive_all_morph_ResNet10_500e/encoder.pt",
         'encoder':Contrastive_Encoder_resnet10,
         "tabular_encoder": Tabular_Encoder,
         "encoder_optim": optim.Adam,
         "encoder_lr": 0.0002,
         "embClassifer": None,
         "loss": nn.BCELoss(), #[NTXentLoss(),nn.BCELoss(), #CLIPLoss(0.1,0.5),#NTXentLoss()]
         "epochs": 50
     } """
    
#     picai_Multmodal_Contrastive_all_morph_ResNet10_500e_CLS_head_3l_head = {
#          "dataset": 'Balanced-picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
#          "toShuffle": False,
#          "toTrain":True,
#          "isMultiModal":False,
#          "selected_features":["patient_age","psa","psad","prostate_volume"], 
#                             #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
#                             #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
#          "isContrastive": False,
#          "useAI_Segmentation":True,
#          "biasFeildCorrection":False,
#          "resample": True,
#          "modalities":["t2w"],
#          "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
#          "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
#          "batchSize": 125,
#          "batchAccumulator": 1,#15
#          "percentage": 1,
#          'encoder_emb_sz':100,
#          'augmentation':augmentation_supervised,
#          'table_corruption': 0.3,
#          'encoder_weights_path':"/data1/practical-sose23/morphometric/results_arno_new/picai/picai_Multmodal_Contrastive_all_morph_ResNet10_500e/encoder.pt",
#          'encoder':Contrastive_Encoder_resnet10,
#          "tabular_encoder": Tabular_Encoder,
#          "encoder_optim": optim.Adam,
#          "encoder_lr": 0.0002,
#          "embClassifer": None,
#          "loss": nn.BCELoss(), #[NTXentLoss(),nn.BCELoss(), #CLIPLoss(0.1,0.5),#NTXentLoss()]
#          "epochs": 100
#      }

    """ picai_Multmodal_Contrastive_lasso_orig_ResNet10_500e  = {
         "dataset": 'picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
         "toShuffle": False,
         "toTrain":True,
         "isMultiModal":True,
         "selected_features":PICAI_Tabular_Features["original+Lasso"],
         "isContrastive": True,
         "useAI_Segmentation":True,
         "biasFeildCorrection":False,
         "resample": True,
         "modalities":["t2w"],
         "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
         "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
         "batchSize": 100,
         "batchAccumulator": 2,#15
         "percentage": 1,
         'encoder_emb_sz':100,
         'augmentation':augmentation_contrastive,
         'table_corruption': 0.3,
         'encoder_weights_path':None,
         'encoder':Contrastive_Encoder_resnet10,
         "tabular_encoder": Tabular_Encoder,
         "encoder_optim": optim.Adam,
         "encoder_lr": 0.002,
         "embClassifer": None,
         "loss": CLIPLoss(0.5,0.5),#CLIPLoss(0.1,0.5),#NTXentLoss(),
         "epochs": 500
    } """

    """ picai_Multmodal_Contrastive_lasso_orig_ResNet10_500e_CLS_head = {
         "dataset": 'Balanced-picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
         "toShuffle": False,
         "toTrain":True,
         "isMultiModal":False,
         "selected_features":["patient_age","psa","psad","prostate_volume"], 
                            #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
                            #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
         "isContrastive": False,
         "useAI_Segmentation":True,
         "biasFeildCorrection":False,
         "resample": True,
         "modalities":["t2w"],
         "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
         "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
         "batchSize": 125,
         "batchAccumulator": 1,#15
         "percentage": 1,
         'encoder_emb_sz':100,
         'augmentation':augmentation_supervised,
         'table_corruption': 0.3,
         'encoder_weights_path':"/data1/practical-sose23/morphometric/results_arno/picai/picai_Multmodal_Contrastive_lasso_orig_ResNet10_500e/encoder.pt",
         'encoder':Contrastive_Encoder_resnet10,
         "tabular_encoder": Tabular_Encoder,
         "encoder_optim": optim.Adam,
         "encoder_lr": 0.0002,
         "embClassifer": None,
         "loss": nn.BCELoss(), #[NTXentLoss(),nn.BCELoss(), #CLIPLoss(0.1,0.5),#NTXentLoss()]
         "epochs": 50
     } """
    
    """ picai_Multmodal_Contrastive_all_morph_ResNet10_500e_corruption02  = {
         "dataset": 'picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
         "toShuffle": False,
         "toTrain":True,
         "isMultiModal":True,
         "selected_features":PICAI_Tabular_Features["morphometric"],
         "isContrastive": True,
         "useAI_Segmentation":True,
         "biasFeildCorrection":False,
         "resample": True,
         "modalities":["t2w"],
         "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
         "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
         "batchSize": 100,
         "batchAccumulator": 1,#15
         "percentage": 1,
         'encoder_emb_sz':100,
         'augmentation':augmentation_contrastive,
         'table_corruption': 0.2,
         'encoder_weights_path':None,
         'encoder':Contrastive_Encoder_resnet10,
         "tabular_encoder": Tabular_Encoder,
         "encoder_optim": optim.Adam,
         "encoder_lr": 0.002,
         "embClassifer": None,
         "loss": CLIPLoss(0.5,0.5),#CLIPLoss(0.1,0.5),#NTXentLoss(),
         "epochs": 500
    } """
    
#     picai_Multmodal_Contrastive_lasso_orig_ResNet10_500e_CLS_head_3l_deep_200e = {
#          "dataset": 'Balanced-picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
#          "toShuffle": False,
#          "toTrain":True,
#          "isMultiModal":False,
#          "selected_features":["patient_age","psa","psad","prostate_volume"], 
#                             #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
#                             #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
#          "isContrastive": False,
#          "useAI_Segmentation":True,
#          "biasFeildCorrection":False,
#          "resample": True,
#          "modalities":["t2w"],
#          "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
#          "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
#          "batchSize": 125,
#          "batchAccumulator": 1,#15
#          "percentage": 1,
#          'encoder_emb_sz':100,
#          'augmentation':augmentation_supervised,
#          'table_corruption': 0.3,
#          'encoder_weights_path':"/data1/practical-sose23/morphometric/results_arno_new/picai/picai_Multmodal_Contrastive_lasso_orig_ResNet10_500e/encoder.pt",
#          'encoder':Contrastive_Encoder_resnet10,
#          "tabular_encoder": Tabular_Encoder,
#          "encoder_optim": optim.Adam,
#          "encoder_lr": 0.0002,
#          "embClassifer": None,
#          "loss": nn.BCELoss(), #[NTXentLoss(),nn.BCELoss(), #CLIPLoss(0.1,0.5),#NTXentLoss()]
#          "epochs": 200
#      } 
    
    """ picai_Multmodal_Contrastive_t2w_ResNet10_500e  = {
         "dataset": 'picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
         "toShuffle": False,
         "toTrain":True,
         "isMultiModal":True,
         "selected_features":PICAI_Tabular_Features["T2W"],
         "isContrastive": True,
         "useAI_Segmentation":True,
         "biasFeildCorrection":False,
         "resample": True,
         "modalities":["t2w"],
         "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
         "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
         "batchSize": 100,
         "batchAccumulator": 2,#15
         "percentage": 1,
         'encoder_emb_sz':100,
         'augmentation':augmentation_contrastive,
         'table_corruption': 0.3,
         'encoder_weights_path':None,
         'encoder':Contrastive_Encoder_resnet10,
         "tabular_encoder": Tabular_Encoder,
         "encoder_optim": optim.Adam,
         "encoder_lr": 0.002,
         "embClassifer": None,
         "loss": CLIPLoss(0.5,0.5),#CLIPLoss(0.1,0.5),#NTXentLoss(),
         "epochs": 500
    } """

#     picai_Multmodal_Contrastive_t2w_ResNet10_500e_CLS_head_3l_deep_200e = {
#          "dataset": 'Balanced-picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
#          "toShuffle": False,
#          "toTrain":True,
#          "isMultiModal":False,
#          "selected_features":["patient_age","psa","psad","prostate_volume"], 
#                             #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
#                             #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
#          "isContrastive": False,
#          "useAI_Segmentation":True,
#          "biasFeildCorrection":False,
#          "resample": True,
#          "modalities":["t2w"],
#          "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
#          "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
#          "batchSize": 125,
#          "batchAccumulator": 1,#15
#          "percentage": 1,
#          'encoder_emb_sz':100,
#          'augmentation':augmentation_supervised,
#          'table_corruption': 0.3,
#          'encoder_weights_path':"/data1/practical-sose23/morphometric/results_arno_new/picai/picai_Multmodal_Contrastive_t2w_ResNet10_500e/encoder.pt",
#          'encoder':Contrastive_Encoder_resnet10,
#          "tabular_encoder": Tabular_Encoder,
#          "encoder_optim": optim.Adam,
#          "encoder_lr": 0.0002,
#          "embClassifer": None,
#          "loss": nn.BCELoss(), #[NTXentLoss(),nn.BCELoss(), #CLIPLoss(0.1,0.5),#NTXentLoss()]
#          "epochs": 100
#      } 
    
    """ picai_Multmodal_Contrastive_adc_ResNet10_500e  = {
         "dataset": 'picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
         "toShuffle": False,
         "toTrain":True,
         "isMultiModal":True,
         "selected_features":PICAI_Tabular_Features["ADC"],
         "isContrastive": True,
         "useAI_Segmentation":True,
         "biasFeildCorrection":False,
         "resample": True,
         "modalities":["t2w"],
         "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
         "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
         "batchSize": 100,
         "batchAccumulator": 2,#15
         "percentage": 1,
         'encoder_emb_sz':100,
         'augmentation':augmentation_contrastive,
         'table_corruption': 0.3,
         'encoder_weights_path':None,
         'encoder':Contrastive_Encoder_resnet10,
         "tabular_encoder": Tabular_Encoder,
         "encoder_optim": optim.Adam,
         "encoder_lr": 0.002,
         "embClassifer": None,
         "loss": CLIPLoss(0.5,0.5),#CLIPLoss(0.1,0.5),#NTXentLoss(),
         "epochs": 500
    } """

#     picai_Multmodal_Contrastive_adc_ResNet10_500e_CLS_head_3l_deep_200e = {
#          "dataset": 'Balanced-picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
#          "toShuffle": False,
#          "toTrain":True,
#          "isMultiModal":False,
#          "selected_features":["patient_age","psa","psad","prostate_volume"], 
#                             #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
#                             #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
#          "isContrastive": False,
#          "useAI_Segmentation":True,
#          "biasFeildCorrection":False,
#          "resample": True,
#          "modalities":["t2w"],
#          "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
#          "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
#          "batchSize": 125,
#          "batchAccumulator": 1,#15
#          "percentage": 1,
#          'encoder_emb_sz':100,
#          'augmentation':augmentation_supervised,
#          'table_corruption': 0.3,
#          'encoder_weights_path':"/data1/practical-sose23/morphometric/results_arno_new/picai/picai_Multmodal_Contrastive_t2w_ResNet10_500e/encoder.pt",
#          'encoder':Contrastive_Encoder_resnet10,
#          "tabular_encoder": Tabular_Encoder,
#          "encoder_optim": optim.Adam,
#          "encoder_lr": 0.0002,
#          "embClassifer": None,
#          "loss": nn.BCELoss(), #[NTXentLoss(),nn.BCELoss(), #CLIPLoss(0.1,0.5),#NTXentLoss()]
#          "epochs": 100
#      }
    

    """ picai_Multmodal_Contrastive_all_features_ResNet10_200e  = {
         "dataset": 'picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
         "toShuffle": False,
         "toTrain":True,
         "isMultiModal":True,
         "selected_features":PICAI_Tabular_Features["all"],
         "isContrastive": True,
         "useAI_Segmentation":True,
         "biasFeildCorrection":False,
         "resample": True,
         "modalities":["t2w"],
         "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
         "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
         "batchSize": 100,
         "batchAccumulator": 2,#15
         "percentage": 1,
         'encoder_emb_sz':100,
         'augmentation':augmentation_contrastive,
         'table_corruption': 0.3,
         'encoder_weights_path':None,
         'encoder':Contrastive_Encoder_resnet10,
         "tabular_encoder": Tabular_Encoder,
         "encoder_optim": optim.Adam,
         "encoder_lr": 0.002,
         "embClassifer": None,
         "loss": CLIPLoss(0.5,0.5),#CLIPLoss(0.1,0.5),#NTXentLoss(),
         "epochs": 200
    } """

    """ picai_Multmodal_Contrastive_all_features_ResNet10_200e_larger_batches_lr1e_3_corr_02  = {
         "dataset": 'picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
         "toShuffle": False,
         "toTrain":True,
         "isMultiModal":True,
         "selected_features":PICAI_Tabular_Features["all"],
         "isContrastive": True,
         "useAI_Segmentation":True,
         "biasFeildCorrection":False,
         "resample": True,
         "modalities":["t2w"],
         "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
         "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
         "batchSize": 190,
         "batchAccumulator": 1,#15
         "percentage": 1,
         'encoder_emb_sz':100,
         'augmentation':augmentation_contrastive,
         'table_corruption': 0.3,
         'encoder_weights_path':None,
         'encoder':Contrastive_Encoder_resnet10,
         "tabular_encoder": Tabular_Encoder,
         "encoder_optim": optim.Adam,
         "encoder_lr": 0.002,
         "embClassifer": None,
         "loss": CLIPLoss(0.5,0.5),#CLIPLoss(0.1,0.5),#NTXentLoss(),
         "epochs": 500
    } """

    picai_Multmodal_Contrastive_all_features_ResNet10_200e_larger_batches_lr1e_3_corr_02_CLS_head_3l_deep = {
         "dataset": 'Balanced-picai',  #['picai','Balanced-picai','cifar-10', 'cifar-100',]
         "toShuffle": False,
         "toTrain":True,
         "isMultiModal":False,
         "selected_features":["patient_age","psa","psad","prostate_volume"], 
                            #["original_gldm_LowGrayLevelEmphasis_T2W","original_gldm_SmallDependenceEmphasis_T2W",
                            #"original_gldm_LowGrayLevelEmphasis_ADC","original_gldm_SmallDependenceEmphasis_ADC"],
         "isContrastive": False,
         "useAI_Segmentation":True,
         "biasFeildCorrection":False,
         "resample": True,
         "modalities":["t2w"],
         "targetVoxelShape": (1.26,1.26,3.78),#(2.0, 2.0, 6.0),
         "targetimageSize":[80,80,20],#[195//2, 195//2, 99/6], #[220, 220, 45],
         "batchSize": 125,
         "batchAccumulator": 1,#15
         "percentage": 1,
         'encoder_emb_sz':100,
         'augmentation':augmentation_supervised,
         'table_corruption': 0.3,
         'encoder_weights_path':"/data1/practical-sose23/morphometric/results_arno_new/picai/picai_Multmodal_Contrastive_all_features_ResNet10_200e_larger_batches_lr1e_3_corr_02/encoder.pt",
         'encoder':Contrastive_Encoder_resnet10,
         "tabular_encoder": Tabular_Encoder,
         "encoder_optim": optim.Adam,
         "encoder_lr": 0.0002,
         "embClassifer": None,
         "loss": nn.BCELoss(), #[NTXentLoss(),nn.BCELoss(), #CLIPLoss(0.1,0.5),#NTXentLoss()]
         "epochs": 100
     }
   
    
    
    
    def __str__(self):
        return self.value


experimentsAll = [Experiment(experimentType=i) for i in ExperimentEnums]
