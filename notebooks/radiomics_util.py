import numpy as np
import matplotlib.pyplot as plt
from radiomics import featureextractor
import os
import SimpleITK as sitk
import math
import pandas as pd


data_path = "/data1/practical-sose23/morphometric/data/"
seg_guer_path = "/data1/practical-sose23/morphometric/picai_labels/anatomical_delineations/whole_gland/AI/Guerbet23/"

data_path_cropped = "/data1/practical-sose23/morphometric/data_cropped/"
seg_guer_path_cropped = "/data1/practical-sose23/morphometric/picai_labels_cropped/anatomical_delineations/whole_gland/AI/Guerbet23/"

def get_image_and_segmentaion(patient_id, study_id, modality = "t2w", use_cropped = True):
    if not isinstance(patient_id, str):
        patient_id = str(patient_id)

    if not isinstance(study_id, str):
        study_id = str(study_id)

    if use_cropped:
        d_path = data_path_cropped
        s_path = seg_guer_path_cropped
    else:
        d_path = data_path
        s_path = seg_guer_path
    
    image = None
    segmentation = None
    for file in os.listdir(d_path + patient_id + "/"):
        string = "{0}_{1}_{2}.mha".format(patient_id,study_id,modality)
        if string == file:
            if study_id in file:
                image = sitk.ReadImage(d_path + patient_id + "/" +  file, sitk.sitkFloat32)
                

    for file in os.listdir(s_path):
        if modality == "t2w" and patient_id+"_"+study_id + ".nii.gz" == file:
            
            segmentation = sitk.ReadImage(s_path + file, sitk.sitkUInt8)

        if modality == "adc" and patient_id+"_"+study_id + ".nii.gz" == file:
            
            segmentation = sitk.ReadImage(s_path + file, sitk.sitkUInt8)

    return (image, segmentation)

def plot_sitk(sitk_obj, mask=None):

    pixel_array = sitk.GetArrayFromImage(sitk_obj)
    
    columns = 5
    rows = math.ceil(pixel_array.shape[0] / columns)

    fig = plt.figure(figsize=(20,rows*4))

    for i in range(pixel_array.shape[0]):
        #im = data[:,:,i]
        #mask = 
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(pixel_array[i], cmap="gray", interpolation="none")
        if mask is not None:
            array = sitk.GetArrayFromImage(mask)
            plt.imshow(array[i], cmap="Reds", alpha = 0.9*(array[i]>0), interpolation="none")


def find_file_in_folder(search_str, path_to_folder):
    for file in os.listdir(path_to_folder):
        if search_str in file:
            return(path_to_folder +  file)
        

def iterate_over_data(function, data_path = "/data1/practical-sose23/morphometric/data/", image_modality="t2w",  limit = np.infty):
    result = []
    for i, file in enumerate(os.listdir(data_path)):

        #print(i)

        if i > limit:
            break

        result.append(function(find_file_in_folder(image_modality, data_path + file + "/")))

    return result


def resample_image(image, mask, desired_voxel_shape = None):
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator = sitk.sitkNearestNeighbor #sitk.sitkLinear

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


# this calculates the number of pixels between the prostate and all borders
def get_border(mask):
    array = sitk.GetArrayFromImage(mask)
    from scipy.ndimage import find_objects

    # Find the bounding box coordinates
    
    bounding_box = find_objects(array)[0]

    # Calculate the padding distances
    padding_z_start = bounding_box[0].start
    padding_z_end = array.shape[0] - bounding_box[0].stop
    padding_x_start = bounding_box[1].start
    padding_x_end = array.shape[1] - bounding_box[1].stop
    padding_y_start = bounding_box[2].start
    padding_y_end = array.shape[2] - bounding_box[2].stop

    # Print the padding distances
    result = {
    "Padding X (Start)": padding_x_start,
    "Padding X (End)": padding_x_end,
    "Padding Y (Start)": padding_y_start,
    "Padding Y (End)": padding_y_end,
    "Padding Z (Start)": padding_z_start,
    "Padding Z (End)": padding_z_end,
    }
    return result

def header_info(s):
    image, mask = get_image_and_segmentaion(s["patient_id"], s["study_id"])
    s["spacing_0"] =  image.GetSpacing()[0]
    s["spacing_1"] = image.GetSpacing()[1]
    s["spacing_2"] = image.GetSpacing()[2]
    s["size_0"] =  image.GetSize()[0]
    s["size_1"] =  image.GetSize()[1]
    s["size_2"] =  image.GetSize()[2]
    s["origin"] = image.GetOrigin()
    s["real_dimension_0"] = image.GetSpacing()[0] * image.GetSize()[0]
    s["real_dimension_1"] = image.GetSpacing()[1] * image.GetSize()[1]
    s["real_dimension_2"] = image.GetSpacing()[2] * image.GetSize()[2]
    s["size_image"] = image.GetSize()
    s["size_mask"] = mask.GetSize()
    borders = get_border(mask)
    s.append(pd.Series(borders))
    return s
        