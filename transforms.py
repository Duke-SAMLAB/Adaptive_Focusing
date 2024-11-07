import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.optim as optim

from common import *

class Focusing_Transform(torch.nn.Module):
  """
  Base class which wraps common functionalities of all focusing transforms.
  """
  def __init__(self, M, f_, device='cpu',flg_trainable=True):
    """
    Focusing_Transform constructor.

    Args:
      M: Number of array elements
      f_: 1D vector of frequencies to perform focusing over
      device: CUDA device type, default is 'cpu'
    """
    super().__init__()
    self.f_ = f_ # Vector of frequencies to focus at
    self.F = len(self.f_) # Number of frequency bins
    self.M = M # Number of array elements
    self.flg_trainable = flg_trainable # Indicates that this focusing transform is trainable
    self.flg_partially_adaptive = False # Let the appropriate classes set this in their constructors
    self.device = device # CUDA device
    self.Tf___ = None # initialize the focusing transformations to None for now 
    self.param_dict = None # Dictionary of trainable parameters, helps in serialization and deserialization
    self.init_parameters()

  def forward(self,x):
    """
    Forward function applies stored focusing transformations on data x.
    Note: This is needed because torch.nn.Module expects a forward function defined
    Returns: 
      Focused data xtilde
    """
    Tf___ =self.get_all_transforms()
    return Tf___ @ x
  
  def init_parameters(self):
    """
    Function stub for parameter initalization
    This is to be filled in by inheriting class
    """
    pass
  
  def get_all_transforms(self):
    """
    Function stub for getting all the transforms
    This is left to the inheriting class to implement for computational reasons
    """
    pass

  def get_transform(self, fidx, nidx=None):
    """
    Gets the transform at a particular index.
    Returns: 
      Tf__ for frequency index fidx
    """
    return self.get_all_transforms()[fidx]

  def get_diagonal_from_weights(self, Dexp__):
    """
    Utility function which returns the set of diagonal matrices Dfq__ given the weights WDq__
    Returns: 
      F x M x M diagonal matrices
    """
    D___ = torch.diag_embed(torch.exp(-1j*Dexp__).T) # F x M x M
    return D___

  def get_householder_from_weights(self, WRp__):
    """
    Utility function which returns the set of Householder matrices Rfp__ given the weights WRp__
    Args:
      WRp__: F x M weight matrix containing the parameters for this given Householder reflection matrices
    Returns: 
      F x M x M Householder matrices
    """
    M = self.M # Number of array elements
    F = self.F # Number of frequency bins
    r__ = WRp__.T / torch.linalg.norm(WRp__.T,ord=2,dim=1,keepdim=True) # F x M
    r___ = r__.unsqueeze(-1)
    eye___ = torch.eye(M,device=self.device).repeat(F,1,1) # F x M x M
    R___ = eye___ - 2*r___ @ H(r___)
    return R___
    
  def get_focused_covariances(self, Rf___):
    """
    Applies all focusing transformations to a set of F x M x M narrowband covariance matrices
    Returns: 
      F x M x M focused covariances
    """
    Tf___ = self.get_all_transforms()
    return Tf___ @ Rf___ @ H(Tf___)
      
  def get_wideband_covariance(self, Rf___):
    """
    Utility function which takes a set of unfocused narrowband covariances, applies focusing, sums and returns
    Args:
      Rf___: F x M x M tensor of the set of unfocused narrowbadn covariances
    Returns:
      Rwb__: M x M focused, wideband covariance
    """
    Rfoc___ = self.get_focused_covariances(Rf___)
    Rwb__ = torch.sum(Rfoc___,axis=0)
    return Rwb__

class Presteer_Transform(Focusing_Transform):
  """
  Pre-steering transformation class for a uniform line array (ULA)
  """
  def __init__(self, M, f_, presteer_bearing, d):
    """
    Constructor.
    Args:
      M: Number of array elements
      f_: 1D vector of frequencies to perform focusing over
      presteer_bearing: Angle to presteer to, measure CCW from along the array axis
      d: ULA spacing in meters
    """
    super().__init__(M, f_)
    self.flg_trainable = False # set to false because this is not a trainable transform
    self.presteer_bearing = presteer_bearing # pre-steering bearing
    self.n_ = np.arange(self.M) # vector of element indices
    self.d = d # element spacing

  def init_parameters(self):
    pass
  
  def get_all_transforms(self):
    """
    Utility function which gets all focusing transformations.
    Returns:
      Tf___: F x M x M set of diagonal focusing matrices
    """
    M = self.M # Number of array elements
    F = self.F # Number of frequency bins
    Tf___ = torch.zeros([F,M,M],dtype=torch.complex128)
    ### Loop through F frequencies
    for fidx in range(F):
      om = 2*np.pi*self.f_[fidx]
      a_ = np.exp(-1j*(om/SOUND_SPEED)*self.n_*self.d*self.presteer_bearing)
      Tf___[fidx] = torch.from_numpy(np.diag(a_))
    return Tf___

class Fully_Adaptive_Focusing(Focusing_Transform):
  """
  Fully adaptive focusing class.
  """
  def init_parameters(self):
    """
    Initializes all parameters of the transformation randomly using torch.randn.
    """
    INIT_VALUE = 1e-3 # Hard-coded initialization scaling
    M = self.M # Number of array elements
    F = self.F # Number of frequency bins
    init_func_Diagonal = torch.randn # the random function to use for the parameters of the diagonal transforms
    init_func_Householder = torch.randn # the random function to use for the parameters of the Householder transforms

    ### Fully-adaptive focusing parameters defined in paper section 3B 
    self.WD1__ = torch.nn.Parameter(INIT_VALUE*init_func_Diagonal(M,F,dtype=torch.double).to(self.device))
    self.WD2__ = torch.nn.Parameter(INIT_VALUE*init_func_Diagonal(M,F,dtype=torch.double).to(self.device))
    self.WD3__ = torch.nn.Parameter(INIT_VALUE*init_func_Diagonal(M,F,dtype=torch.double).to(self.device))
    self.WR4__ = torch.nn.Parameter(INIT_VALUE*init_func_Householder(M,F,dtype=torch.complex128).to(self.device))
    self.WR3__ = torch.nn.Parameter(INIT_VALUE*init_func_Householder(M,F,dtype=torch.complex128).to(self.device))
    self.WR2__ = torch.nn.Parameter(INIT_VALUE*init_func_Householder(M,F,dtype=torch.complex128).to(self.device))
    self.WR1__ = torch.nn.Parameter(INIT_VALUE*init_func_Householder(M,F,dtype=torch.complex128).to(self.device))

    ### Fixed DFT and IDFT matrices
    self.DFT__ = (1/np.sqrt(self.M))*torch.from_numpy(scipy.linalg.dft(M)).to(self.device) # M x M
    self.IDFT__ = H(self.DFT__) # M x M

  def init_parameters_presteer(self, k_, dsinth):
    """
    Initializes all parameters so that the resulting transformation is equivalent to pre-steering
    Args:
      k_: Horizontal wavenumbers at each frequency bin. Should just be 2*np.pi * f_ / C in free-space
      dsinth: d * sin(bearing). Input the array spacing * sin(bearing), using broadside as 0
    """
    M = self.M # Number of array elements
    F = self.F # Number of frequency bins

    # Set all the parameters other than those for D3 to zero
    self.WD1__.data = torch.zeros([M,F],dtype=torch.double).to(self.device)
    self.WD2__.data = torch.zeros([M,F],dtype=torch.double).to(self.device)
    
    self.WR4__.data = torch.zeros([M,F],dtype=torch.complex128).to(self.device)
    self.WR3__.data = torch.zeros([M,F],dtype=torch.complex128).to(self.device)
    self.WR2__.data = torch.zeros([M,F],dtype=torch.complex128).to(self.device)
    self.WR1__.data = torch.zeros([M,F],dtype=torch.complex128).to(self.device)
    
    self.WR4__.data[0,:] = 1
    self.WR3__.data[0,:] = 1
    self.WR2__.data[0,:] = 1
    self.WR1__.data[0,:] = 1
    
    D1___ = self.get_diagonal_from_weights(self.WD1__) # F x M x M
    D2___ = self.get_diagonal_from_weights(self.WD2__) # F x M x M
    D3___ = self.get_diagonal_from_weights(self.WD3__) # F x M x M

    R4__ = self.get_householder_from_weights(self.WR4__) # F x M x M
    R3__ = self.get_householder_from_weights(self.WR3__) # F x M x M
    R2__ = self.get_householder_from_weights(self.WR2__) # F x M x M
    R1__ = self.get_householder_from_weights(self.WR1__) # F x M x M
        
    DFT___ = self.DFT__.repeat(F,1,1) # F x M x M
    IDFT___ = self.IDFT__.repeat(F,1,1) # F x M x M

    I___ = R4__ @ R3__ @ IDFT___ @ D2___ @ R2__ @ R1__ @ DFT___ @ D1___
    
    eye = torch.eye(M).repeat(F,1,1).to(self.device)
    assert(torch.all(torch.abs(R4__ @ R3__ - eye) < 1e-4))
    assert(torch.all(torch.abs(R2__ @ R1__ - eye) < 1e-4))
    assert(torch.all(torch.abs(D2___ - eye) < 1e-4))
    assert(torch.all(torch.abs(D1___ - eye) < 1e-4))
    assert(torch.all(torch.abs(I___ - eye) < 1e-4))

    for kidx,k in enumerate(k_):
      self.WD3__.data[:,kidx] = torch.arange(M)*k*dsinth

  def get_all_transforms(self):
    """
    Utility function which returns all the focusing transformations.
    This is GPU optimized, as the matrix multiplications will occur on the specified device
    Returns:
      Tf___: F x M x M set of M x M focusing transformations using current parameters
    """
    M = self.M # Number of array elements
    F = self.F # Number of frequency bins

    D1___ = self.get_diagonal_from_weights(self.WD1__) # F x M x M
    D2___ = self.get_diagonal_from_weights(self.WD2__) # F x M x M
    D3___ = self.get_diagonal_from_weights(self.WD3__) # F x M x M

    R4__ = self.get_householder_from_weights(self.WR4__) # F x M x M
    R3__ = self.get_householder_from_weights(self.WR3__) # F x M x M
    R2__ = self.get_householder_from_weights(self.WR2__) # F x M x M
    R1__ = self.get_householder_from_weights(self.WR1__) # F x M x M
        
    DFT___ = self.DFT__.repeat(F,1,1) # F x M x M
    IDFT___ = self.IDFT__.repeat(F,1,1) # F x M x M

    Tf___ = D3___ @ R4__ @ R3__ @ IDFT___ @ D2___ @ R2__ @ R1__ @ DFT___ @ D1___

    return Tf___ # F x M x M

class Partially_Adaptive_Focusing(Focusing_Transform):
  """
  Partially adaptive focusing.
  """
  def __init__(self, M, f_,**kwargs):
    """
    Constructor.
    Args:
      M: Number of array elements
      f_: 1D vector of frequencies to focus over
    """
    super().__init__(M, f_,**kwargs)
    self.flg_partially_adaptive = True # This is used when training

  def init_parameters(self):
    pass

  def init_parameters_from_fully_adaptive(self, transform,N):
    """
    Utility function which intializes the transform based on a pre-trained fully-adaptive one
    Args:
      transform: Pre-trained Fully_Adaptive_Focusing object
      N: Hyperparameter to set the dimension of U and V. Should be set such that N < MF / (M+F)
    """
    ### Use a trained fully-adaptive transform to initialize the partially adaptive transform
    self.UD3__.data,self.VD3__.data = factorize_weight_matrix(transform.WD3__,N)
    self.UD2__.data,self.VD2__.data = factorize_weight_matrix(transform.WD2__,N)
    self.UD1__.data,self.VD1__.data = factorize_weight_matrix(transform.WD1__,N)
    
    self.UR4__.data,self.VR4__.data = factorize_weight_matrix(transform.WR4__,N)
    self.UR3__.data,self.VR3__.data = factorize_weight_matrix(transform.WR3__,N)
    self.UR2__.data,self.VR2__.data = factorize_weight_matrix(transform.WR2__,N)
    self.UR1__.data,self.VR1__.data = factorize_weight_matrix(transform.WR1__,N)
    
  def init_parameters_set_N(self,parameter_reduction_factor):
    """
    Utility function which randomly initializes the weight matrices.
    Args:
      parameter_reduction_factor: Defines the factor by which to REDUCE the number of parameters. Set to 4 in the paper
    """
    INIT_VALUE = 1e-3 # Hard-coded initialization scaling
    M = self.M # Number of array elements
    F = self.F # Number of frequency bins

    ### Compute N given the parameter reduction factor
    bound_on_N = (M * F) // (M + F)
    N = bound_on_N // parameter_reduction_factor
    self.N = N
    # Obviously, if parameter reduction factor is too great, N=1, this doesn't make sense for focusing!
    assert(N >= 2)

    init_func = torch.randn
    # factorization of D3
    self.UD3__ = torch.nn.Parameter(INIT_VALUE*init_func(M,N,dtype=torch.double).to(self.device)) # Random
    self.VD3__ = torch.nn.Parameter(INIT_VALUE*init_func(N,F,dtype=torch.double).to(self.device)) # row of ones, then small values

    # factorization of D2
    self.UD2__ = torch.nn.Parameter(INIT_VALUE*init_func(M,N,dtype=torch.double).to(self.device))
    self.VD2__ = torch.nn.Parameter(INIT_VALUE*init_func(N,F,dtype=torch.double).to(self.device))

    # factorization of D3
    self.UD1__ = torch.nn.Parameter(INIT_VALUE*init_func(M,N,dtype=torch.double).to(self.device))
    self.VD1__ = torch.nn.Parameter(INIT_VALUE*init_func(N,F,dtype=torch.double).to(self.device))

    # factorization of R1
    self.UR4__ = torch.nn.Parameter(INIT_VALUE*init_func(M,N,dtype=torch.complex128).to(self.device))
    self.VR4__ = torch.nn.Parameter(INIT_VALUE*init_func(N,F,dtype=torch.complex128).to(self.device))

    # factorization of R1a
    self.UR3__ = torch.nn.Parameter(INIT_VALUE*init_func(M,N,dtype=torch.complex128).to(self.device))
    self.VR3__ = torch.nn.Parameter(INIT_VALUE*init_func(N,F,dtype=torch.complex128).to(self.device))

    # factorization of R2
    self.UR2__ = torch.nn.Parameter(init_func(M,N,dtype=torch.complex128).to(self.device))
    self.VR2__ = torch.nn.Parameter(init_func(N,F,dtype=torch.complex128).to(self.device))

    # factorization of R2a
    self.UR1__ = torch.nn.Parameter(init_func(M,N,dtype=torch.complex128).to(self.device))
    self.VR1__ = torch.nn.Parameter(init_func(N,F,dtype=torch.complex128).to(self.device))

    self.DFT__ = (1/np.sqrt(self.M))*torch.from_numpy(scipy.linalg.dft(M)).to(self.device)
    self.IDFT__ = H(self.DFT__)
    
  def init_parameters_presteer(self, k_, dsinth):
    """
    Utility function which initializes the partially-adaptive focusing as a pre-steering transformation
    """
    M = self.M # Number of array elements
    F = self.F # Number of frequency bins
    N = self.N # Number of rows in UDq, VDq, URp, VRp

    small_init_value = 1e-3
    ### D3 should result in the pre-steering transformation
    dk = k_[1] - k_[0] # assume wavenumbers are equally spaced...
    self.UD3__.data[:,0] = torch.arange(M)*k_[0]*dsinth #self.A1__.data[:,0] # leave this alone as a random constant
    self.UD3__.data[:,1] = torch.arange(M)*dk*dsinth # this is our pre-steer phasing, (delta om/c) * ndsinth
    self.UD3__.data[:,2:] = small_init_value*torch.randn([M,N-2],dtype=torch.double)
    self.VD3__.data[0,:] = 1
    self.VD3__.data[1,:] = 1*torch.arange(F)
    self.VD3__.data[2:,:] = small_init_value

    
    self.VD2__.data[1:,:] = 1
    self.VD2__.data[0,:] = 1
    self.VD1__.data[1:,:] = 1
    self.VD1__.data[0,:] = 1

    ### Set so that R4 = R3 and R2 = R1
    self.UR3__.data = self.UR4__.data
    self.VR3__.data = self.VR4__.data
    self.UR1__.data = self.UR2__.data
    self.VR1__.data = self.VR2__.data

    ### For inspection purposes, set Df1, Df2, and Df3 = Identity for f index 0
    D1___ = self.get_diagonal_from_weights(self.UD1__ @ self.VD1__) # F x M x M
    D1___ = H(D1___[0]).repeat(F,1,1) @ D1___
    D2___ = self.get_diagonal_from_weights(self.UD2__ @ self.VD2__)
    D2___ = H(D2___[0]).repeat(F,1,1) @ D2___
    D3___ = self.get_diagonal_from_weights(self.UD3__ @ self.VD3__)
    D3___ = H(D3___[0]).repeat(F,1,1) @ D3___

    R1___ = self.get_householder_from_weights(self.UR1__ @ self.VR1__) # F x M x M
    R2___ = self.get_householder_from_weights(self.UR2__ @ self.VR2__) # F x M x M
    R3___ = self.get_householder_from_weights(self.UR3__ @ self.VR3__) # F x M x M
    R4___ = self.get_householder_from_weights(self.UR4__ @ self.VR4__) # F x M x M
        
    DFT___ = self.DFT__.repeat(F,1,1) # F x M x M
    IDFT___ = self.IDFT__.repeat(F,1,1) # F x M x M

    I___ = R4___ @ R3___ @ IDFT___ @ D2___ @ R2___ @ R1___ @ DFT___ @ D1___
    
    eye = torch.eye(M).repeat(F,1,1).to(self.device)
    assert(torch.all(torch.abs(R4___ @ R3___ - eye) < 1e-4))
    assert(torch.all(torch.abs(R2___ @ R1___ - eye) < 1e-4))
    assert(torch.all(torch.abs(D1___ - eye) < 1e-4))
    assert(torch.all(torch.abs(D2___ - eye) < 1e-4))
    assert(torch.all(torch.abs(I___ - eye) < 1e-4))

  def get_all_transforms(self):
    """
    Utility function which returns all focusing transformations.
    This function is GPU optimized by calling PyTorch's batch matrix multiply
    Returns:
      Tf___: F x M x M set of M x M focusing transformations using current parameters
    """
    M = self.M # Number of array elements
    F = self.F # Number of frequency bins
    WD1__ = self.UD1__ @ self.VD1__ # M x F = M x N @ N x F
    WD2__ = self.UD2__ @ self.VD2__ # M x F = M x N @ N x F
    WD3__ = self.UD3__ @ self.VD3__ # M x F = M x N @ N x F

    D1___ = self.get_diagonal_from_weights(WD1__) # F x M x M
    D2___ = self.get_diagonal_from_weights(WD2__) # F x M x M
    D3___ = self.get_diagonal_from_weights(WD3__) # F x M x M

    WR1__ = self.UR1__ @ self.VR1__ # M x F = M x N @ N x F
    WR2__ = self.UR2__ @ self.VR2__ # M x F = M x N @ N x F
    WR3__ = self.UR3__ @ self.VR3__ # M x F = M x N @ N x F
    WR4__ = self.UR4__ @ self.VR4__ # M x F = M x N @ N x F

    R1___ = self.get_householder_from_weights(WR1__) # F x M x M
    R2___ = self.get_householder_from_weights(WR2__) # F x M x M
    R3___ = self.get_householder_from_weights(WR3__) # F x M x M
    R4___ = self.get_householder_from_weights(WR4__) # F x M x M
    
    DFT___ = self.DFT__.repeat(F,1,1) # F x M x M
    IDFT___ = self.IDFT__.repeat(F,1,1) # F x M x M

    Tf___ = D3___ @ R4___ @ R3___ @ IDFT___ @ D2___ @ R2___ @ R1___ @ DFT___ @ D1___
    return Tf___

from torch.optim.lr_scheduler import ReduceLROnPlateau
import gc

def train_transform(
  transform, Rx___, lr, num_epochs, device='cpu',
  lf=1e-3, min_lr=1e-6, min_improvement=1e-3,min_epochs=32, flg_print=True):
  """
  Primary training function, which takes in training hyperparameters
  Args:
    transform: Focusing_Transform object to train
    Rx___: F x M x M narrowband sample covariances from the data
    lr: Learning rate, a good default value is 1e-3
    num_epochs: Number of epochs to train over. Typically set to 1024 or 2048.
    device: CUDA device string, defaults to 'cpu'
    lf: Loading factor applied to the narrowband covariances. This minimizes sensitivity to noise during training.
    min_improvement: Minimum reduction in loss function before early stopping.
    min_epochs: Minimum number of epochs before early stopping criteria are evaluated
  """

  ### SETUP ###
  if not transform.flg_trainable:
    print("train_transform called on data independent transform, exitting")
    return
  transform.to(device) # Move transform and parameters to specified device
  F,M,_ = Rx___.shape
  Rx___ = Rx___.to(device) # Move data to specified device
  Rxlf___ = torch.zeros(Rx___.shape,dtype=torch.complex128).to(device) # Loaded covariances
  with torch.no_grad():
    eye___ = (lf*batch_trace(Rx___).unsqueeze(-1).unsqueeze(-1)*torch.eye(M).repeat(F,1,1).to(device))
    Rxlf___ = (Rx___ + eye___).detach()

  optimizer = optim.Adam(transform.parameters(), lr=lr)  # Updated to use .parameters()
  scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
  
  loss_ = []
  logdet_ = []
  loss_prev = torch.tensor(1e99).to(device)

  ### MAIN TRAINING LOOP ###
  for epoch in range(num_epochs):
    optimizer.zero_grad()
    Rwb__ = transform.get_wideband_covariance(Rxlf___)

    ### LOSS CALCULATION AND BACKPROPAGATION ###
    loss = torch.real(torch.logdet(Rwb__))
    rwb__ = transform.get_wideband_covariance(Rx___)
    logdet = torch.real(torch.logdet(rwb__)).cpu().detach().numpy().item()
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    current_lr = optimizer.param_groups[0]['lr']

    if flg_print:
      print(f'Epoch [{epoch+1}/{num_epochs}], LogDet(Rwb): {logdet}, Loss: {loss.cpu().item()}')
    
    ### save values from current iteration ###
    loss_.append(loss.item())
    logdet_.append(logdet)

    ### EARLY STOPPING CRITERIA ###
    if epoch > min_epochs:
      if current_lr < min_lr:
        print("early stopping due to minimum learning rate: %1.4f"%current_lr)
        break
      improvement = np.abs(loss.item() - loss_prev.item())
      if improvement < min_improvement:
        print("early stopping due to small improvement: %1.4f"%improvement)
        print("loss:%1.4f loss_prev: %1.4f"%(loss,loss_prev))
        break
    loss_prev = loss

  try:
    del(Rxlf___)
    del(Rx___)
    del(Rwb__)
    gc.collect()
    torch.cuda.empty_cache()
  except:
     print("Couldn't clear variables, moving on")
  return loss_,logdet_