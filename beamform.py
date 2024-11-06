import numpy as np
import scipy.signal as sgn
import importlib
import os
import pickle
from scipy import signal
from tqdm import tqdm
import gc
# Check if PyTorch is installed
torch_spec = importlib.util.find_spec("torch")
if torch_spec is not None:
    import torch

import transforms as tf
from common import *

def get_src_points_1D(array_x__,depth, max_az=90,N=128,rbf=1e9):
  maxsinth = np.sin(np.radians(max_az))
  sinth_ = np.linspace(-1*maxsinth,maxsinth,N)
  th_ = np.arccos(sinth_)
  x_ = rbf*np.cos(th_)
  y_ = rbf*np.sin(th_)
  z_ = depth*np.ones(len(x_))
  src_x__ = np.asarray([x_,y_,z_]).transpose()
  return th_,src_x__

def get_src_points_far_field(array_x__,beams_per_el=8,rbf=1e9):
  num_el = array_x__.shape[0]
  N = beams_per_el * num_el
  sinth_ = np.linspace(-1,1,N)
  th_ = np.arcsin(sinth_)
  x_ = rbf*np.cos(np.pi/2 - th_)
  y_ = rbf*np.sin(np.pi/2 - th_)
  src_x__ = np.asarray([x_,y_]).transpose()
  return th_,src_x__

def get_steering_vector(array_x__,src_x_,f):
  M = array_x__.shape[0]
  om = 2*np.pi*f
  prop_x__ = array_x__ - np.kron(src_x_,np.ones([M,1]))
  a_ = np.exp(-1j*(om/swp.SWELLEX_SOUND_SPEED)*np.linalg.norm(prop_x__,axis=1)).reshape([M,1])
  a_ *= np.conj(a_[0])
  a_ /= np.linalg.norm(a_)
  return a_

def get_wideband_steering_vectors(array_x__,src_x_,f_):
  F = len(f_)
  M = array_x__.shape[0]
  A__ = np.zeros([F,M],dtype=np.complex128)
  for fidx,f in enumerate(f_):
    A__[fidx,:] = get_steering_vector(array_x__,src_x_,f).flatten()
  return A__

def get_steering_matrix(array_x__,src_x__,f):
  M = array_x__.shape[0]
  N = src_x__.shape[0]
  om = 2*np.pi*f
  A__ = np.zeros([N,M],dtype=np.cfloat)
  for idx,src_x_ in enumerate(src_x__):
    a_ = get_steering_vector(array_x__,src_x_,f)
    A__[idx,:] = a_.flatten()
  return A__.transpose()

def get_wideband_steered_power(W___, R___,log=False):
  Nfreq,Nel,Nbeam = W___.shape
  if torch_spec is not None and torch.cuda.is_available():
    bf_device = torch.device("cuda")  # Use CUDA device if available
    W___ = W___.to(bf_device)
    R___ = R___.to(bf_device)
    sp__ = torch.abs(cmn.extract_diagonals(H(W___) @ R___ @ W___)).cpu().numpy()
  else: # drop back to using numpy and loop over frequency and beams
    print("Unable to run beamformer on GPU, falling back to CPU")
    sp__ = np.zeros([Nfreq,Nbeam])
    for fidx in range(Nfreq):
      W__ = W___[fidx]
      sp__[fidx,:] = np.abs(np.diag(H(W__) @ R___[fidx] @ W__).flatten())
  return sp__

def get_all_weights_CBF(V___, Rf___, Tfoc___):
  Vfoc___ = Tfoc___ @ V___
  W___ = H(Tfoc___) @ Vfoc___ # should be identity, but useful to make sure transformations are working as expected
  return W___

def get_all_weights_narrowband_MVDR(V___, Rf___, Tfoc___, load_factor=0):
  F, M, Nbeams = V___.shape
  Rfloaded___ = torch.zeros(Rf___.shape,dtype=Rf___.dtype)
  for fidx in range(F):
    Rfloaded___[fidx] = load_matrix_simple(Rf___[fidx],load_factor)
  Rfinv___ = torch.linalg.inv(Rfloaded___)
  Wnum___ = Rfinv___ @ V___  # F x M x Nbeams = F x M x M @ F x M x Nbeams
  Wdenom___ = cmn.extract_diagonals(torch.abs(H(V___) @ Rfinv___ @ V___)) # F x N = diag( F x N x M @ F x M x M @ F x M x N)
  Wdenom___ = Wdenom___.unsqueeze(1) # F x 1 x N
  W___ = Wnum___ / (Wdenom___.expand(F,M,Nbeams))
  return W___

def get_all_weights_wideband_MVDR(V___, Rf___, Tfoc___, freq_weight_ = None, load_factor=0, flg_norm_trace=True,flg_W_denom=True):
  F, M, Nbeams = V___.shape
  Vfoc___ = Tfoc___ @ V___
  Rfoc___ = Tfoc___ @ Rf___ @ H(Tfoc___)

  ### Apply frequency weighting and normalization according to kwargs ###
  if freq_weight_ is None:
    freq_weight_ = torch.ones(F)
  freq_weight___ = freq_weight_.view(F,1,1)
  if flg_norm_trace:
    Rfoc___ /= batch_trace(Rfoc___).view(F,1,1)
  Rfoc___ *= freq_weight___

  Rwb__ = torch.mean(Rfoc___,axis=0).detach()
  Rfpp___ = torch.unsqueeze(Rwb__,0).repeat(F,1,1)

  Rfpploaded___ = torch.zeros_like(Rfpp___)
  for fidx in range(F):
    Rfpploaded___[fidx] = load_matrix_simple(Rfpp___[fidx], load_factor)
  Rfppinv___ = torch.linalg.inv(Rfpploaded___)
  Wnum___ = Rfppinv___ @ Vfoc___  # F x M x Nbeams = F x M x M @ F x M x Nbeams
  if flg_W_denom:
    Wdenom___ = cmn.extract_diagonals(torch.abs(H(Vfoc___) @ Rfppinv___ @ Vfoc___)) # F x N = diag( F x N x M @ F x M x M @ F x M x N)
  else:
    Wdenom___ = torch.ones([F,Nbeams])
  Wdenom___ = Wdenom___.unsqueeze(1) # F x 1 x N
  W___ = Wnum___ / (Wdenom___.expand(F,M,Nbeams))
  return W___