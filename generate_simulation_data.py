import numpy as np
import torch
import pickle
import os
import json
import argparse
from os.path import expanduser
from copy import deepcopy

from common import SWELLEX_SOUND_SPEED, SWELLEX_CHANNEL_DEPTH, H
import beamform as bf
import common as cmn
import transforms as tf
import simulation as sim

if __name__ == '__main__':
  print("Running main driver script which generates simulation data, performs focusing, and beamforms.")
  # Initialize argparse
  parser = argparse.ArgumentParser(description="Process commandline arguments.")
  parser.add_argument('config_path', type=str, help='Path to the configuration JSON file.')
  args = parser.parse_args()
  config_path = args.config_path

  device = "cuda" if torch.cuda.is_available() else "cpu"
  try:
    with open(config_path,'r') as f:
      config = json.load(f)
  except Exception as e:
    print("Could not find or load a valid simulation configuration at %s"%config_path)
    raise(e)

  ### Environment parameters ###
  m_f0 = config['m_f0'] # starting frequency
  m_bandwidth = config['m_bandwidth'] # total bandwidth in Hz
  m_df = config['m_df'] # frequency spacing in Hz
  m_flg_src_on_bin = True
  m_flg_array_error = False
  m_Nsnaps = config['m_Nsnaps'] # number of snapshots to simulate
  m_Nbeams = config['m_Nbeams'] # number of beams in grid, used to ensure source is placed on grid

  ### Array parameters ###
  m_int_snr_ = np.asarray(config['m_int_snr_'])
  m_M_HLA = config['m_M_HLA']
  m_HLA_z0 = config['m_HLA_z0']
  m_M = m_M_HLA
  m_target_SNR = config['m_target_snr']
  m_target_tone_spacing = config['m_target_tone_spacing']
  m_flg_target_planewave = config.pop('m_flg_target_planewave',1)

  ### SNR measurement parameters
  ### These are used to control how SNR is measured.
  ### For wideband sources and channels with fading, SNR should be measured across the band
  ### For tonal sources, do not average across the band, set this flag to 0!
  ### m_flg_normalize_a is used to determine whether the returned source wavefront should be normalized or not
  m_flg_snr_avg = config['m_flg_snr_avg'] # are the SNRs given average or in-band?
  m_flg_normalize_a = not m_flg_snr_avg # if snr is given as averaged across the whole band, don't normalize

  ### Target and interference parameters ###
  m_int_range_bearing_depth_ = np.asarray(config['m_int_range_bearing_depth']) # meters, degrees, meters 
  m_target_range_bearing_depth_ = np.asarray(config['m_target_range_bearing_depth']) # meters, degrees, meters 

  ### Load processing parameters ###
  LEARNING_RATE = 1e-3
  NUM_EPOCHS = config['p_num_epochs'] # How many epochs to run training for
  focusing_lf = config['p_focusing_load_factor'] # How much to load the focused wideband covariance. Sets the sensitivity to noise eigenvalues
  parameter_reduction_factor = config['p_parameter_reduction_factor'] # Parameter reduction factor (Letter O in paper) for partially-adaptive focusing
  flg_init_presteer = config['p_flg_init_presteer'] # initialize focusing transforms as a pre-steering transform?
  p_min_improvement = config['p_min_improvement'] # minimum improvement in cost function before early stopping
  p_min_lr = config['p_min_lr'] # minimum learning rate before early stopping
  p_min_epochs = config['p_min_epochs'] # minimum number of epochs to run training for, regardless of early stopping criteria

  ### Computed values from config parameters ###
  m_fmax = m_f0 + m_bandwidth # top of frequency band in Hz
  m_F = int(m_bandwidth / m_df) # number of frequency bins
  m_f_ = m_f0 + m_df*np.arange(m_F) # vector of all frequencies
  m_fcenter = np.mean(m_f_) # center frequency
  m_lam = (SWELLEX_SOUND_SPEED / m_fmax) # lambda, used to determine array spacing. avoid grating lobes, use lambda corresponding to highest frequency
  m_d = m_lam/2 # array spacing chosen to be critically sampled
  m_D = SWELLEX_CHANNEL_DEPTH # ~SwellEx depth
  m_beamwidth = 2 / m_M_HLA # compute the array beamwidth
  sinth_ = np.linspace(-1,1,m_Nbeams) # used to grid up azimuth space, ensure target is on a beam

  ### Define array X,Y, and Z positions ###
  x_centered_ = m_d*np.arange(-m_M_HLA/2,m_M_HLA/2) # 
  x_zerod_ = x_centered_ - np.min(x_centered_)
  m_array_x__ = np.asarray([x_centered_,np.zeros(m_M_HLA),np.ones(m_M_HLA)*m_HLA_z0]).transpose()

  ### Variables holding all source parameters ###
  m_Nint = m_int_range_bearing_depth_.shape[0] # number of interferers
  m_r_ = np.ones(m_Nint+1) # vector to store all source ranges
  m_th0_ = np.zeros(m_Nint+1) # vector to store all source bearings
  m_z_ = np.ones(m_Nint+1) # vector to store all source depths
  m_snr__ = np.ones([m_Nint+1,m_F]) # vector to store all source SNRs at all frequencies
  m_L = m_Nint+1 # number of sources = number of interferers + 1 target

  ### Interference parameters ###
  m_int_idx_ = np.asarray(np.arange(m_Nint)) # indices of interferers
  m_int_snr__ = np.kron(m_int_snr_.reshape([m_Nint,1]).T,np.ones([m_F,1])).T # SNRs of interference across the band
  m_rint_ = m_int_range_bearing_depth_[:,0] # range of all interferers
  m_thint_ = m_int_range_bearing_depth_[:,1] # bearing of all interferers
  m_zint_ = m_int_range_bearing_depth_[:,2] # depth of all interferers

  ### Load up the interference parameters ###
  m_th0_[m_int_idx_] = np.radians(m_thint_) # 
  m_r_[m_int_idx_] = m_rint_
  m_z_[m_int_idx_] = m_zint_
  m_snr__[m_int_idx_,:] = m_int_snr__

  ### Target parameters ###
  m_src_idx = m_Nint # target index
  m_rsrc = m_target_range_bearing_depth_[0]
  m_thsrc = m_target_range_bearing_depth_[1]
  m_zsrc = m_target_range_bearing_depth_[2]

  ### Load up the target parameters ###
  m_th0_[m_src_idx] = np.radians(m_thsrc) # target bearing
  m_r_[m_src_idx] = m_rsrc # target range
  m_z_[m_src_idx] = m_zsrc
  m_snr__[m_src_idx,:] = -50 
  m_snr__[m_src_idx,::m_target_tone_spacing] = m_target_SNR

  m_src_x0__ = np.vstack((m_r_*np.cos(m_th0_),m_r_*np.sin(m_th0_),m_z_)).T

  m_data_gen_params = dict(
    src_x0__ = m_src_x0__,
    array_x__ = m_array_x__,
    f_ = m_f_,
    D = m_D,
    snr__ = m_snr__,
    flg_normalize_a = m_flg_normalize_a,
  )
  
  Xs____ = np.zeros([m_F,m_L+1,m_M,m_Nsnaps],dtype=np.cfloat)

  ### GENERATE SIMULATION DATA ###
  m_A___ = sim.get_model_A(m_data_gen_params)
  for f in np.arange(m_F):
    sigma_ = np.power(10,m_snr__[:,f]/10) / m_M
    S__ = np.diag(sigma_)
    Rsl___ = [] # list of rank-1 signal covariances
    for l in range(m_L): # loop over L sources
      a_ = m_A___[f,l,:].reshape([m_M,1])
      a_ /= np.linalg.norm(a_)
      Rsl___.append(sigma_[l] * a_ @ H(a_))
      Xs____[f,l,:,:] = sim.gen_data_from_R(Rsl___[l],m_Nsnaps)

    ### Generate the noise signals ###
    N__ = (1/m_M)*np.eye(m_M)
    Xs____[f,-1,:,:] = sim.gen_data_from_R(N__,m_Nsnaps)
    
  ### SCALE COVARIANCES ALL THE SAME###
  Rl____ = sim.form_covariances(Xs____)
  ### DON'T DO THIS WHEN USING A TONAL TARGET ###
  if m_flg_snr_avg:
    print("Averaging SNR over the band")
    for l in range(m_L):
      SNR = sim.avg_SNR_from_covariance(Rl____[l],log=False) / sim.avg_SNR_from_covariance(Rl____[-1],log=False)
      set_snr = m_target_SNR if l == m_src_idx else m_int_snr_[l]
      print(10**(set_snr/10)/m_M)
      scaling = np.sqrt((1/SNR)*10**(set_snr/10)/m_M)
      Xs____[:,l,:,:] *= scaling
    
  Sf___,If___,Nf___ = sim.form_SIN_covariances(Xs____,m_src_idx,m_int_idx_,-1)
  Rf___ = Sf___ + If___ + Nf___
  print("After normalizing data:")
  print("Output SNR: %1.1f dB"%(10*np.log10(m_M*sim.avg_SNR_from_covariance(Sf___,log=False)/sim.avg_SNR_from_covariance(Nf___,log=False))))
  print("Output INR: %1.1f dB"%(10*np.log10(m_M*sim.avg_SNR_from_covariance(If___,log=False)/sim.avg_SNR_from_covariance(Nf___,log=False))))
  print("NNR: %1.1f dB"%(10*np.log10(sim.avg_SNR_from_covariance(Nf___,log=False)/sim.avg_SNR_from_covariance(Nf___,log=False))))

  X___ = np.sum(Xs____,axis=1)
  targ_th_deg = m_target_range_bearing_depth_[1]
  int_th_deg = m_int_range_bearing_depth_[-1,1]
  sinth = np.cos(np.radians(int_th_deg))
  print("Pre-steering to direction of nearby interference at %f degrees" % int_th_deg)

  dsinth = m_d * sinth
  _transforms = {
    #'diag':tf.Diagonal_Transform(m_M,m_f_,device=device),
    'Pre-steer': tf.Presteer_Transform(m_M,m_f_,sinth,m_d),
    'Fully_Adaptive': tf.Fully_Adaptive_Focusing(m_M, m_f_, device=device),
    'Partially_Adaptive': tf.Partially_Adaptive_Focusing(m_M,m_f_,device=device),
  }

  ### INITIALIZE ADAPTIVE TRANFORMATIONS ###
  _transform_params_ = {} # dictionary which will store the changing params
  _transform_train_times = {}
  for key,transform in _transforms.items():
    transform.init_parameters()
    _transform_params_[key] = []
    _transform_train_times[key] = []
    if transform.flg_partially_adaptive:
      transform.init_parameters_set_N(parameter_reduction_factor)
    if flg_init_presteer:
      if isinstance(transform,tf.Fully_Adaptive_Focusing) or isinstance(transform,tf.Partially_Adaptive_Focusing):
        print("Initializing %s as a pre-steering transform"%key)
        k_ = 2*np.pi*m_f_ / cmn.SOUND_SPEED
        transform.init_parameters_presteer(k_,dsinth)
        Tf___ = transform.get_all_transforms().to(device)
        Tf___ = H(Tf___[0]).repeat(m_F,1,1) @ Tf___

  Xproc___ = torch.from_numpy(X___) # Narrowband data to be processed
  Rx___ = Xproc___ @ H(Xproc___) # Narrowband covariances

  ### Train transforms and save data ###
  for key,transform in _transforms.items():
    #Note: we initialize the Partially Adaptive params from the Fully Adaptive params
    if isinstance(transform, tf.Partially_Adaptive_Focusing):
      N = transform.UD1__.shape[1]
      transform.init_parameters_from_fully_adaptive(_transforms['Fully_Adaptive'],N)
    if transform.flg_trainable:
      print(f'Training transform: {key}')
      loss_,logdet_ = tf.train_transform(
        transform, Rx___,
        LEARNING_RATE,
        NUM_EPOCHS,
        device=device,
        flg_print=False,
        lf=focusing_lf,
        min_lr=p_min_lr,
        min_improvement=p_min_improvement,
        min_epochs=p_min_epochs)
      transform.loss_ = loss_
      transform.logdet_ = logdet_
    transform.param_dict = transform.state_dict()
    _transform_params_[key].append(deepcopy(transform.state_dict()))


  ### BEAMFORMING ###

  ### Beamformers are defined by a tuple, a weight computation function, it's arguments, and a flag indicated if it should be applied to focused data or not
  WB_MVDR_load_factor = 1e-3
  NB_MVDR_load_factor = 1e-3
  _beamformers = {
    'Conventional': (bf.get_all_weights_CBF,dict(),False), # CBF weights, no arguments, use unfocused data
    'Narrowband MVDR lf=%.2e'%NB_MVDR_load_factor: (bf.get_all_weights_narrowband_MVDR,dict(load_factor=NB_MVDR_load_factor),False),
    'Wideband MVDR lf=%.2e'%(WB_MVDR_load_factor): (bf.get_all_weights_wideband_MVDR,dict(load_factor=WB_MVDR_load_factor,flg_norm_trace=False),True),
  }

  device = 'cpu' # peform beamforming back on CPU
  
  ### Load beamforming processing parameters from configuration ###
  Nbeams = config["p_bf_nbeams"] # number of beams to process
  th0deg = config["p_bf_th0_deg"] # starting bearing
  th1deg = config["p_bf_th1_deg"] # ending bearing
  rbf = m_target_range_bearing_depth_[0] # range to beamform to
  thdeg_ = np.linspace(th0deg,th1deg,num=Nbeams) # 1D vector of bearings, in degrees CCW from array axis
  th_ = np.radians(thdeg_) # 1D vector of bearings, in radians, CCW from array axis
  src_x__ = np.vstack((rbf*np.cos(th_),rbf*np.sin(th_),m_D*np.ones(Nbeams))).T # X,Y positions to beamform to, allows for near-field
  V___ = torch.from_numpy(np.asarray([bf.get_steering_matrix(m_array_x__,src_x__,f) for f in m_f_])).to(device) # F x M x Nbeams contains all replica vectors

  output_path = os.getcwd()

  ### Outputs to be saved ###
  _fraz___ = {} # nested dictionary, transform --> beamformer --> FRAZ___
  ### Begin beamforming ###
  with torch.no_grad():
    for tf_name,transform in _transforms.items():
      print("*** Beamforming for transform %s ***"%tf_name)
      _fraz___[tf_name] = {}
      for bfkey in _beamformers.keys():
        _fraz___[tf_name][bfkey] = np.zeros([m_F,Nbeams])

      Tf___ = transform.get_all_transforms().to(device)
      for Tf__ in Tf___:
        assert(cmn.is_unitary(Tf__))

      Rxfoc___ = Tf___ @ Rx___ @ H(Tf___)
      Rwb__ = torch.sum(Rxfoc___,0).detach()

      for bfkey,beamformer_tup in _beamformers.items():
        print("Running beamformer %s"%bfkey)
        weight_func, weight_kwargs,flg_focus_data = beamformer_tup # unpack tuple
        W___ = weight_func(V___, Rx___, Tf___,**weight_kwargs).to(device) # Compute weights
        Rbf___ = Rxfoc___ if flg_focus_data else Rx___ # Covariance to use in w^H R w
        Vbf___ = Tf___ @ V___ if flg_focus_data else V___ # Replica vectors to use
        fraz__ = bf.get_wideband_steered_power(W___,Rbf___,log=False) # Compute steered power, F x Nbeams
        _fraz___[tf_name][bfkey] = fraz__ # Save resuts to dictionary

  ### Save results to a pickle file for external plotting
  with(open(os.path.join(os.getcwd(),'fraz_results.pkl'),mode='wb')) as f:
    pickle.dump(_fraz___,f)