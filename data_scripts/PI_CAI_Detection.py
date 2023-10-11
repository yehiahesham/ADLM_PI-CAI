import sys
sys.path.append('../')

import os,json,random,sys,math
import os.path as osp
import matplotlib.pyplot as plt
from typing import List, Tuple
import copy
import torch
#import torchvision
#import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
# import cv2
import numpy as np
import pandas as pd
import numpy as np
import SimpleITK as sitk
from PIL import Image

labels_path='/data1/practical-sose23/morphometric/picai_labels'
canerSeg_HumanExp  = labels_path+"/csPCa_lesion_delineations/human_expert/resampled"





class PI_CAI_Detection(data.Dataset):
    def __init__(self, image_path,labels_path,modalities=["t2w"],\
        multiModal=True,useAI_Segmentation=True,biasFeildCorrection=False, normalization = True,\
        resample=False, targetVoxelShape=(1,1,3),targetimageSize=(220,220,45),\
        transform=None, selected_features=[],corruption=None):
        '''
        image_path: scan images data path 
        labels_path: segmentation path to patients scan
        modalities : list which modalities to load TODO: current impl.supports only 1 modality 
            loading all modalities short form =["all"] instead of  ["t2w","hbv","adc",...]
            Possible Modalities:
                Axial T2-weighted imaging (t2w)
                Axial high b-value (≥ 1000 s/mm2) diffusion-weighted imaging (hbv)
                Axial apparent diffusion coefficient maps (adc)  
                Sagittal T2-weighted imaging(sag)			            
                Coronal T2-weighted imaging(cor)
        targetVoxelShape : target spacing to std all scan images
        t2w_features : Features to select from the t2w radiomics
        adc_features : Features to select from the adc radiomics
        '''
        self.root               = image_path
        self.labels_path        = labels_path
        self.table_path         = labels_path+"/clinical_information/marksheet_filled_simple_normalized.csv" #marksheet_filled_complex_normalized.csv #marksheet_filled_normalized.csv
        self.glandSeg_path      = labels_path+"/anatomical_delineations/whole_gland/AI/Guerbet23"
        self.canerSeg_HumanExp  = labels_path+"/csPCa_lesion_delineations/human_expert/resampled"
        self.canerSeg_AI        = labels_path+"/csPCa_lesion_delineations/AI/Bosma22a"
        self.useAI_Segmentation  = useAI_Segmentation
        if self.useAI_Segmentation: 
            self.canerSeg_path = self.canerSeg_AI
        else: 
            self.canerSeg_path = self.canerSeg_HumanExp

        if "all" in modalities: modalities= ["adc","t2w","hbv","sag","cor"]
        self.modalities=modalities
        self.multiModal=multiModal
        self.targetimageSize=targetimageSize
        self.targetVoxelShape=targetVoxelShape
        self.resample = resample
        self.biasFeildCorrection=biasFeildCorrection
        self.normalization = normalization
        self.transform=transform
        self.selected_features=selected_features
        self.corruption = corruption

        self.clincical_info = pd.read_csv(self.table_path)
        # self.tw2_radiomics = pd.read_pickle("/data1/practical-sose23/morphometric/t2w_radiomic_features.pkl")
        # self.adc_radiomics = pd.read_pickle("/data1/practical-sose23/morphometric/adc_radiomic_features.pkl")

        self.prepocess()
        self.generate_marginal_distributions()

        self.StudyIds  = self.clincical_info['study_id'].unique() 
        self.PatientIds = self.clincical_info['patient_id'].unique()
        
    def __len__(self):
        return len(self.StudyIds)

    def __getitem__(self, index):    
        # scan,case_csPCa,glandSeg,cancerSeg,patient_id,study_id,\
        #     patient_age,psa,psad,prostate_volume, t2w, adc = self.pull_item(index)
        # table_feat = (patient_age,psa,psad,prostate_volume) + tuple(t2w) + tuple(adc)
        
        scan,case_csPCa,glandSeg,cancerSeg,patient_id,study_id,\
            table_feat = self.pull_item(index)
        
        view_1, view_2 = self.generate_imaging_views(scan, 0) #scan, augmented scan
        if self.multiModal:
            return scan,view_2,case_csPCa,tuple(table_feat)   
        else :
            return view_1, view_2, case_csPCa
     
    def generate_imaging_views(self, scan, augmentation_rate):
        """
        Generates two views of the scan for unimodal contrastive learning. 
        The first is always augmented. The second has {augmentation_rate} chance to be augmented.
        """
        view_2 = self.transform(scan)
        if random.random() < augmentation_rate:
            view_1 = self.transform(scan)
        else:
            view_1 = scan
        return view_1, view_2

    def prepocess(self):
        # self.clincica|l_info["mri_date"] =  [x.date() for x in pd.to_datetime(self.clincical_info["mri_date"])]
        self.clincical_info["case_csPCa"] =self.clincical_info["case_csPCa"].replace({'YES': 1, 'NO': 0})
        # self.clincical_info["case_csPCa"] = self.clincical_info["case_csPCa"].astype('int')
        # df =df.replace({'subscribed': {'yes': True, 'no': False}}) #multiple replace at the same time.
        
        # if to train with human expert labeling,
        # drop study ids with no human expert labeling 
        if not self.useAI_Segmentation:
            files = os.listdir(canerSeg_HumanExp)
            files = [fn.replace(".nii.gz", "") for fn in files if ".nii.gz" in fn and "._" not in fn]
            human_labled_subject_ids = ["_".join(fn.split("_")[0:2]) for fn in files]
            human_labled_subject_ids = sorted(list(set(human_labled_subject_ids)))
            human_labled_subject_ids = [int(subject_id.split('_')[1]) for subject_id in human_labled_subject_ids]
            self.clincical_info = self.clincical_info[self.clincical_info['study_id'].isin(human_labled_subject_ids)]
            

    
    
    def get_studyInfo(self,studyId):
        return self.clincical_info.loc[self.clincical_info['study_id'] == studyId]
    
    def get_patientInfo(self,patientId):
        return self.clincical_info.loc[self.clincical_info['patient_id'] == patientId]
    
    def get_Lables(self):
        return self.clincical_info['case_csPCa']
    
    def get_segmentations(self,patient_id,study_id):
        gland,cancer = None,None
        seg_path = "{0}/{1}_{2}.nii.gz".format(self.glandSeg_path,patient_id,study_id)
        scan = sitk.ReadImage(seg_path)
        gland=scan
        
        seg_path = "{0}/{1}_{2}.nii.gz".format(self.canerSeg_path,patient_id,study_id)
        scan = sitk.ReadImage(seg_path)
        cancer=scan

        return gland,cancer
    
    def get_scans(self,patient_id,study_id):
        scans=[]
        for modality in self.modalities:
            modality_path = "{0}/{1}/{1}_{2}_{3}.mha".format(self.root,patient_id,study_id,modality)
            scan = sitk.ReadImage(modality_path)
            
            scans.append(scan)
        return scans
    
    def plot(self,img):
        pixel_array = img.numpy()    
        # print(pixel_array.min(), pixel_array.max(), pixel_array.mean())

        columns = 5
        rows = math.ceil(pixel_array.shape[0] / columns)
        fig = plt.figure(figsize=(50,rows*10))

        for i in range(pixel_array.shape[0]):
            fig.add_subplot(rows, columns, i+1)
            plt.imshow(pixel_array[i], cmap="gray")#, interpolation="none")

    def plot_histogram(self, sitk_image):
        
        np_image = sitk.GetArrayFromImage(sitk_image).flatten()
        # print(np_image.min(), np_image.max(), np_image.mean(), np_image.max()/np_image.mean())
        # Plot the histogram of intensities using matplotlib
        plt.hist(np_image, bins=256)
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
        plt.title("Histogram of MRI scan")
        plt.show()
    
    def resample_image(self,image, gland,cancer, targetimageSize=None,targetVoxelShape = None, padding=True):
        if targetimageSize is None: targetimageSize= image.GetSize()
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator = sitk.sitkLinear

        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetOutputDirection(image.GetDirection())

        if targetVoxelShape is None:
            resample.SetSize([int(i) for i in targetimageSize]) 
            resample.SetOutputSpacing(image.GetSpacing())
            resampled_image = image
        
        else:
            if padding:
                resample.SetSize([int(i) for i in targetimageSize])
            else : 
                resample.SetSize([int(image.GetSize()[i] * image.GetSpacing()[i] / targetVoxelShape[i]) for i in range(3)]) 
            resample.SetOutputSpacing(targetVoxelShape)
            resampled_image = resample.Execute(image)
        
        resampled_gland  = None #resample.Execute(gland)
        resampled_cancer = None #resample.Execute(cancer)
        
        return resampled_image,resampled_gland,resampled_cancer
    
    def apply_bias_field_correction(self,image, mask):    
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected_image = corrector.Execute(image, mask)
        #log_bias_field = corrector.GetLogBiasFieldAsImage(image)
        #corrected_image_full_resolution = image / sitk.Exp(log_bias_field)
        
        return corrected_image #, log_bias_field

    def normalize(self, sitk_image):
        

        pixel_array = sitk.GetArrayFromImage(sitk_image)

        min_intensity = pixel_array.min()
        max_intensity = pixel_array.max()
        normalized_array = (pixel_array-min_intensity)/(max_intensity-min_intensity)
        # Convert the normalized numpy array to a sitk object
        normalized_image = sitk.GetImageFromArray(normalized_array)

        # Optionally, set the origin, spacing, and direction if necessary
        normalized_image.SetOrigin(sitk_image.GetOrigin())
        normalized_image.SetSpacing(sitk_image.GetSpacing())
        normalized_image.SetDirection(sitk_image.GetDirection())

        
        return normalized_image
    

    def generate_marginal_distributions(self):
        """
        Generates empirical marginal distribution by transposing data
        """
        data_df = pd.read_csv(self.table_path)
        self.marginal_distributions = data_df.transpose().values.tolist()

    def corrupt(self, subject: List[float]) -> List[float]:
        """
        Creates a copy of a subject, selects the indices 
        to be corrupted (determined by hyperparam corruption_rate)
        and replaces their values with ones sampled from marginal distribution
        """
        subject = copy.deepcopy(subject)

        indices = random.sample(list(range(len(subject))), int(len(subject)*self.corruption)) 
        for i in indices:
            subject[i] = random.sample(self.marginal_distributions[i],k=1)[0] 
        
        return subject

    def pull_item(self, index):
        table_feat  = self.clincical_info.iloc[index]
        table_feat_corrupted = self.corrupt(table_feat) #augment the table
        
        #get Required data
        patient_id       = table_feat["patient_id"].astype(int) 
        study_id         = table_feat["study_id"].astype(int) 
        case_csPCa       = table_feat["case_csPCa"].astype(np.float32)
        
        # get tabular feat
        selected_features= table_feat_corrupted[self.selected_features]
            
        #get scans and segmentation
        scans = self.get_scans(patient_id,study_id)
        glandSeg, cancerSeg = None,None # self.get_segmentations(patient_id,study_id)

        if self.normalization: 
                scans[0] = self.normalize(scans[0])
        if self.resample: 
            #BUG: looks like a simple bug is here.
            scans[0],glandSeg, cancerSeg = self.resample_image(scans[0], glandSeg,cancerSeg, self.targetimageSize,self.targetVoxelShape)
        # if self.biasFeildCorrection: self.apply_bias_field_correction(scans[0],glandSeg)
        
        scans[0]  = sitk.GetArrayFromImage(scans[0]).astype(np.float32)
        scans[0]  = torch.from_numpy(scans[0])
        #glandSeg  = torch.from_numpy(sitk.GetArrayFromImage(glandSeg).astype(np.float32))
        #cancerSeg = torch.from_numpy(sitk.GetArrayFromImage(cancerSeg).astype(np.float32))

        return  scans[0],case_csPCa,glandSeg,cancerSeg,\
                patient_id,study_id,\
                selected_features