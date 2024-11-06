import numpy as np
import pickle
import beamform as bf
from common import SWELLEX_CHANNEL_DEPTH, SWELLEX_SOUND_SPEED, H

def form_covariances(Xs____):
  F,L,M,Nsnaps = Xs____.shape
  R_ = np.empty([L,F,M,M],dtype=np.cfloat)
  for l in range(L):
    for f in range(F):
      R_[l,f,:,:] = (1/Nsnaps)*(Xs____[f,l] @ H(Xs____[f,l]))
  return R_


def form_SIN_covariances(Xs____,signal_idx,int_idx_,noise_idx):
  R_ = form_covariances(Xs____)
  Sf___ = R_[signal_idx]
  If___ = np.sum(R_[int_idx_],axis=0)
  Nf___ = R_[noise_idx]
  return Sf___, If___, Nf___

def SNR_from_covariance(R___,log=True):
  out = np.abs(np.trace(R___,axis1=1,axis2=2))
  if log:
    out = 10*np.log10(out)
  return out

def avg_SNR_from_covariance(R___,log=True):
  snr = np.abs(np.trace(np.sum(R___,axis=0),axis1=0,axis2=1))
  if log:
    return 10*np.log10(snr)
  else:
    return snr

def gen_data_from_R(R__,Nsnaps):
  M = R__.shape[0]
  U__,S_,V__ = np.linalg.svd(R__)
  L__ = U__ @ np.diag(np.sqrt(S_))
  X__ = np.zeros([M,Nsnaps],dtype=np.cfloat)
  for n in range(Nsnaps):
    x_ = (1/np.sqrt(2))*(
      np.random.normal(size=M).reshape([M,1]) + 1j*np.random.normal(size=M).reshape([M,1]))
    X__[:,n] = (L__ @ x_).flatten()
  return X__

def get_model_A(model_params):
  src_x0__ = model_params['src_x0__'] # L x 3 (x,y,z)
  array_x__ = model_params['array_x__'] # M x 3 (x,y,z)
  snr__ = model_params['snr__'] # L x F
  D = model_params['D'] # ocean bottom depth
  f_ = model_params['f_']

  L = src_x0__.shape[0]
  M = array_x__.shape[0]
  F = len(f_)
  A___ = np.zeros([F,L,M],dtype=np.cfloat)
  for l,src_x_ in enumerate(src_x0__):
    A__ = bf.get_wideband_steering_vectors(array_x__,src_x_,f_)
    A___[:,l,:] = A__ / np.mean(np.linalg.norm(A__,axis=1,keepdims=True))
  return A___
