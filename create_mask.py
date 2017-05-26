import nibabel
import numpy as np
import numpy.ma as ma
from functools import reduce

#cd to image directory

#load the image (do this for each image)
clean_even = nibabel.load('PD_m.nii.gz')
clean_even_data = clean_even.get_data()
clean_even_data_mask = np.zeros(clean_even_data.flatten().shape)
clean_even_data_mask[np.where(clean_even_data.flatten() != 0)[0]] = 1.


clean_odd = nibabel.load('PE_m.nii.gz')
clean_odd_data = clean_odd.get_data()
clean_odd_data_mask = np.zeros(clean_odd_data.flatten().shape)
clean_odd_data_mask[np.where(clean_odd_data.flatten() != 0 )[0]] = 1.

novel_even = nibabel.load('ND_m.nii.gz')
novel_even_data = novel_even.get_data()
novel_even_data_mask = np.zeros(novel_even_data.flatten().shape)
novel_even_data_mask[np.where(novel_even_data.flatten()!= 0)[0]] = 1.

novel_odd = nibabel.load('NE_m.nii.gz')
novel_odd_data = novel_odd.get_data()
novel_odd_data_mask = np.zeros(novel_odd_data.flatten().shape)
novel_odd_data_mask[np.where(novel_odd_data.flatten() != 0)[0]] = 1.

all_data_mask = np.logical_and(np.logical_and (clean_even_data_mask,clean_odd_data_mask) , np.logical_and(novel_even_data_mask , novel_odd_data_mask))
print len(np.where(all_data_mask == 1.)[0])
all_data_mask = np.asarray([float(x) for x in all_data_mask])
all_data_mask = all_data_mask.reshape(novel_odd_data.shape)
print all_data_mask.shape
#make a nifti image
print len(np.where(all_data_mask == 0.)[0])
all_mask_nifti = nibabel.Nifti1Image(all_data_mask,affine=clean_even.get_affine())

#save the nifti image
nibabel.save(all_mask_nifti, 'dummy.nii.gz')