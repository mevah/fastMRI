import h5py
import numpy as np
import matplotlib.pyplot as plt
import fastmri
from fastmri.data import transforms as T
from fastmri.data.subsample import EquispacedMaskFunc
import os

#define read and write paths
train_path = "/itet-stor/himeva/bmicdatasets-originals/Originals/fastMRI/brain/multicoil_train"
val_path = "/itet-stor/himeva/bmicdatasets-originals/Originals/fastMRI/brain/multicoil_val"

save_rec_path = "/itet-stor/himeva/net_scratch/fastMRI/recs/"
save_k_path = "/itet-stor/himeva/net_scratch/fastMRI/kspace/"

#create writing paths if they are nonexistent
if not os.path.exists(save_rec_path):
    os.makedirs(save_rec_path)
    #create writing paths if they are nonexistent
if not os.path.exists(save_k_path):
    os.makedirs(save_k_path)


#get train img list
print("Starting getting files from train directory.")
file_list = os.listdir(train_path)

#read and write each slice into a .png file in each scan
for file in file_list:
    words = file.split("_")
    modality = words[2]
    if modality == "AXFLAIR":

        with h5py.File(os.path.join(train_path, file), "r") as hf:
            k = hf["kspace"]
            numslices = hf["kspace"].shape[0]
            print("File:", file)
            for slc in range(numslices):

                #define save path of reconstruction and the sampled kspace
                s_p = os.path.join(save_rec_path, "fullres", file[:-3] + "_" +  str(slc) + ".png")
                s_k = os.path.join(save_k_path, "fullres",
                                   file[:-3] + "_" + str(slc) + ".png")

                # Convert from numpy array to pytorch tensor
                slice_kspace2 = T.to_tensor(k[slc])
                # Apply Inverse Fourier Transform to get the complex image
                slice_image = fastmri.ifft2c(slice_kspace2)
                slice_image_abs = fastmri.complex_abs(slice_image)

                ####fully sampled k space root sum of square reconstruction
                slice_image_rss = fastmri.rss(slice_image_abs, dim=0)   
                plt.imsave(s_p, np.abs(slice_image_rss.numpy()), cmap='gray')
                np.save(s_k, slice_kspace2)

                ###iterate over different accelerations and save the sliced kspaces and reconstructions from them into subfolders
                for acc in range(2,9): 
                    mask_func = EquispacedMaskFunc(center_fractions=[0.04], accelerations=[acc])  # Create the mask function object
                    masked_kspace, mask = T.apply_mask(
                        slice_kspace2, mask_func)   # Apply the mask to k-space
                    # Apply Inverse Fourier Transform to get the complex image
                    sampled_image = fastmri.ifft2c(masked_kspace)
                    # Compute absolute value to get a real image
                    sampled_image_abs = fastmri.complex_abs(sampled_image)
                    sampled_image_rss = fastmri.rss(sampled_image_abs, dim=0)

                    s_p = os.path.join(save_rec_path, "acc"+ str(acc)+ "x",
                                    file[:-3] + "_" + str(slc) + ".png")
                    s_k = os.path.join(save_k_path,  "acc" + str(acc) + "x",
                                    file[:-3] + "_" + str(slc) + ".npy")

                    plt.imsave(s_p, np.abs(sampled_image_rss.numpy()), cmap='gray')
                    np.save(s_k, masked_kspace)
                    print("Saved slice ", str(slc)+ " of acceleration "+ str(acc)+ "x.")
       
