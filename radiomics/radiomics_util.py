import numpy as np
import matplotlib.pyplot as plt
from radiomics import featureextractor
import os
import SimpleITK as sitk
import math


data_path = "/data1/practical-sose23/morphometric/data/"
seg_guer_path = "/data1/practical-sose23/morphometric/picai_labels/anatomical_delineations/whole_gland/AI/Guerbet23/"
seg_bosma_path = "/data1/practical-sose23/morphometric/picai_labels/anatomical_delineations/whole_gland/AI/Bosma22b/"

def get_image_and_segmentaion(patient_id, study_id):
    if not isinstance(patient_id, str):
        patient_id = str(patient_id)

    if not isinstance(study_id, str):
        study_id = str(study_id)
    
    image = None
    segmentation = None
    for file in os.listdir(data_path + patient_id + "/"):
        if "t2w" in file:
            if study_id in file:
                image = sitk.ReadImage(data_path + patient_id + "/" +  file, sitk.sitkFloat32)

    for file in os.listdir(seg_guer_path):
        if patient_id+"_"+study_id in file:
            
            segmentation = sitk.ReadImage(seg_guer_path + file, sitk.sitkUInt8)

    return (image, segmentation)

def plot_sitk(sitk_obj):

    pixel_array = sitk.GetArrayFromImage(sitk_obj)
    
    columns = 5
    rows = math.ceil(pixel_array.shape[0] / columns)

    fig = plt.figure(figsize=(20,rows*4))

    for i in range(pixel_array.shape[0]):
        #im = data[:,:,i]
        #mask = 
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(pixel_array[i], cmap="gray", interpolation="none")


def find_file_in_folder(search_str, path_to_folder):
    for file in os.listdir(path_to_folder):
        if search_str in file:
            return(path_to_folder +  file)
        

def iterate_over_data(function, data_path = "/data1/practical-sose23/morphometric/data/", image_modality="t2w",  limit = np.infty):
    result = []
    for i, file in enumerate(os.listdir(data_path)):

        print(i)

        if i > limit:
            break

        result.append(function(find_file_in_folder(image_modality, data_path + file + "/")))

    return result


def resample_image(image, mask, desired_voxel_shape = None):
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator = sitk.sitkLinear

    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())

    if desired_voxel_shape is None:
        resample.SetSize(image.GetSize()) 
        resample.SetOutputSpacing(image.GetSpacing())
        resampled_image = image
    
    else:
        resample.SetSize([int(image.GetSize()[i] * image.GetSpacing()[i] / desired_voxel_shape[i]) for i in range(3)]) 
        resample.SetOutputSpacing(desired_voxel_shape)
        resampled_image = resample.Execute(image)


    #resample.SetOutputOrigin(mask.GetOrigin())
    #resample.SetOutputDirection(mask.GetDirection())

    resampled_mask = resample.Execute(mask)

    return resampled_image, resampled_mask


def apply_bias_field_correction(image, mask):
    
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(image, mask)
    #log_bias_field = corrector.GetLogBiasFieldAsImage(image)
    #corrected_image_full_resolution = image / sitk.Exp(log_bias_field)
    
    return corrected_image #, log_bias_field


def get_high_correlation_features(df, correlation_threshold = 0.2):
    correlation_matrix = df.corr()

    # Extract the correlation between features and the target variable
    correlation_with_target = correlation_matrix['target'].abs()

    # Select features with correlation above the threshold
    selected_features = correlation_with_target[correlation_with_target > correlation_threshold].index

    # Print the selected features
    #print("Number Selected features:", len(selected_features))
    #print("Selected features:", selected_features)
    
    return(selected_features)