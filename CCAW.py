#!/usr/bin/env python
# encoding: utf-8
"""
CCAW.py

Created by McKell Carter on 2012-03-22.
Modified by Lawrence Ngo throughout 2012.
Copyright (c) 2012 All rights reserved.
"""

import sys
#sys.path.append("/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/") # This allows for the appropriate import of matplotlib on Law's mac.
import os
import unittest
import numpy as N
import nibabel # Pynifti is no longer supported, and it is not supported under the NiBabel package.
import scipy.odr
import matplotlib as mpl
#import pylab as plt
import matplotlib.cm as cm
import matplotlib.path as path
from matplotlib import patches
from matplotlib import collections
from matplotlib import pyplot
from scipy import stats
from matplotlib.font_manager import FontProperties


## CCAW - Cognitive Contrast Analyzer Whole-brain

# BaseData can be either three dimensional of four dimensional.
class BaseData:
	def __init__(self, image_pname, mask_pname):
		self.image_pname = image_pname
		self.mask_pname = mask_pname
		self.image = nibabel.load(self.image_pname)
		self.image_data = self.image.get_data()
		self.mask = nibabel.load(self.mask_pname)
		self.mask_data = self.mask.get_data() != 0
		print('Dimensions of the image are '+str(self.image_data.shape))
		print('Dimensions of the mask are '+str(self.mask_data.shape))
		if (self.image_data.ndim==3) & (self.mask_data.ndim==3):
			self.masker()

	def masker(self):
		if self.image_data.shape==self.mask_data.shape:
			self.image_data_masked = self.image_data[self.mask_data]
			# data = self.mask.get_data().flatten()
			# index = N.where(data==0.)[0]
			# data_mask = self.image_data.flatten()
			# data_mask = [x for i,x in enumerate(data_mask) if i not in index]
			# self.image_data_masked = N.asarray(data_mask)
		else:
			print('The dimensions of the image and the mask are not the same!')
			sys.exit()

# Instantiation creates a list of three-dimensional images sliced along the fourth dimension.
class Data4D(BaseData):
    def __init__(self, image_pname, mask_pname):
    	BaseData.__init__(self, image_pname, mask_pname)
        if self.image_data.ndim != 4:
            print('The image is not a four dimensional file!')
            sys.exit()
        self.slicer()
        self.mask4D()

    # Creates a three-dimensional array for each slice of the four dimensional file and add to self.list_3D
    def slicer(self):
    	self.list_3D = [] # establish a list to insert three-dimensional images.
    	# iterate along the 4th dimension (last dimension):
    	for slice_num in range(self.image_data.shape[-1]):
    		slice = self.image_data[:,:,:,slice_num] # make a slice along the fourth dimension.
    		self.list_3D.append(slice) # Append this to the list.

    # Masks every item in list of 3D arrays. Slicer must have already been run and created self.list_3D
    def mask4D(self):
    	self.image_data_3D_masked_list = []
    	for image_data_3D in self.list_3D:
    		image_data_masked = image_data_3D[self.mask_data]
    		self.image_data_3D_masked_list.append(image_data_masked)

class Analysis3D:
	def __init__(self, x, y, out_pname):
		self.x = x
		self.y = y
		# If x and y are both three dimensional arrays, then run the following.
		if (self.x.image_data.ndim==3) & (self.y.image_data.ndim==3):
			self.run_linear_odr_on_subj_mean(self.x.image_data_masked, self.y.image_data_masked)
			self.plot_subj_mean(self.x.image_data_masked, self.y.image_data_masked, out_pname)
		# If x and y are both four dimensional arrays, then make a plot of each slice along 4th dimension.
		else:
			print('The inputs to analysis are not three dimensional!')
			sys.exit()

	#odr fit and plot residuals
	def f(self, B, x):
		return B[0]*x + B[1]

	# Inputs should be masked 3D numpy arrays.
	def run_linear_odr_on_subj_mean(self, x, y, slope0=0., intercept0=0., expected_x=0., expected_y=0.):
		linear = scipy.odr.Model(self.f)  #make model object from the eq above
		odr_data = scipy.odr.RealData(x.reshape(-1)-expected_x, y.reshape(-1)-expected_y)  #data must be a single dimension (flat) - this is what reshape(-1) does
		self.odr_model = scipy.odr.odrpack.ODR(odr_data, linear, beta0=[slope0,intercept0])
		self.odr_out =  self.odr_model.run()
		self.odr_slope, self.odr_intercept = self.odr_out.beta    #slope and intercept values from odr output
		self.odr_predY = self.odr_slope*x.reshape(-1) + self.odr_intercept
		self.odr_predX = (1./self.odr_slope)*y.reshape(-1) - (self.odr_intercept/self.odr_slope)
		return self.odr_slope, self.odr_intercept

	# Again, inputs should be masked 3D numpy arrays.
	def plot_subj_mean(self, x, y, out_fname='', density_cmap=cm.spectral_r, density_alpha=0.5):
		pyplot.autoscale(enable=False)
		self.fig = plt.figure(figsize=(3., 3.), dpi=150.)
		self.ax = self.fig.add_axes([0.1, 0.125, 0.75, 0.75])#, axisbg='w')
		self.ax_cb = self.fig.add_axes([0.87, 0.15, 0.025, 0.72])
		self.ax.xaxis.set_ticks_position('bottom')
		self.ax.yaxis.set_ticks_position('left')
		this_max = N.max(N.array(N.max(x.reshape(-1)), N.max(y.reshape(-1))))
		this_min = N.min(N.array(N.min(x.reshape(-1)), N.min(y.reshape(-1))))
		self.ax.plot(N.arange(this_min,this_max), N.arange(this_min,this_max), 'b:')    #add 1 to 1 line
		self.ax.plot(N.arange(this_min,this_max), self.odr_slope*N.arange(this_min,this_max)+self.odr_intercept, 'g--', lw=1.5, alpha=0.9) #plot regression
		self.ax.plot(N.arange(this_min,this_max), N.zeros(len(N.arange(this_min,this_max))), 'k--', lw=0.5, alpha=0.9)    #add y = 0
		self.ax.plot(N.zeros(len(N.arange(this_min,this_max))), N.arange(this_min,this_max), 'k--', lw=0.5, alpha=0.9)    #add x = 0
	    #plot 2d-density
		cmap_min = 1.  #cmap_min = N.min(sorted_map_improve_array)
		cmap_max = 100.   #cmap_max = N.max(sorted_map_improve_array)
		cmap_range = cmap_max - cmap_min
		cmap_norm = mpl.colors.Normalize(vmin=cmap_min, vmax=cmap_max)
		cmap =  density_cmap #cm.spectral_r

		self.ax.hexbin(x.reshape(-1), y.reshape(-1), norm=cmap_norm, cmap=density_cmap, alpha=density_alpha, mincnt=1)   #vmin=1, vmax=100,
		self.cb = mpl.colorbar.ColorbarBase(self.ax_cb, cmap=cmap, norm=cmap_norm, orientation='vertical')

		odr_txt_summary = 'odr slope: %.3f, se: %.3f \nodr intercept: %.3f, se: %.3f'  % (self.odr_slope, self.odr_out.sd_beta[0], self.odr_intercept, self.odr_out.sd_beta[1])
		odr_fp = FontProperties(family='Arial', weight='normal', size=8)   #tick properties
#		self.fig.text(0.3, 0.2, odr_txt_summary, fontproperties=odr_fp) #write ODR summary

		#import pdb; pdb.set_trace()

		if len(out_fname)>0:
			self.fig.savefig(out_fname, dpi=300)
			print("Results have been printed to "+ out_fname)
		else:
			self.fig.show()

class Analysis4DSlices(Analysis3D):
	def __init__(self, x, y, out_pname):
		if (x.image_data.ndim==4) & (y.image_data.ndim==4):
			self.slice_odr_slope_intercept = []
			for slice_num in range(len(x.list_3D)):
				out_pname_slice = out_pname + "_" + str(slice_num)
				self.slice_odr_slope_intercept.append(self.run_linear_odr_on_subj_mean(x.list_3D[slice_num], y.list_3D[slice_num]))
				self.plot_subj_mean(x.list_3D[slice_num], y.list_3D[slice_num], out_pname_slice)
		else:
			print('The inputs to analysis are not four dimensional!')
			sys.exit()

class Analysis4DMeans(Analysis3D):
	def __init__(self, x, y, out_pname):
		if (x.image_data.ndim==4) & (y.image_data.ndim==4):
			self.xmean=N.mean(x.image_data, -1)
			self.ymean=N.mean(y.image_data, -1)
			self.xmean_masked = self.xmean[x.mask_data]
			self.ymean_masked = self.ymean[y.mask_data]
			self.run_linear_odr_on_subj_mean(self.xmean_masked, self.ymean_masked)
			self.plot_subj_mean(self.xmean_masked, self.ymean_masked, out_pname)
		else:
			print('The inputs to analysis are not four dimensional!')
			sys.exit()



