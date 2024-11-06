import numpy as np
import torch
SWELLEX_SOUND_SPEED = SOUND_SPEED = 1500
SWELLEX_CHANNEL_DEPTH = 220

def factorize_weight_matrix(W__,N):
  U__,S_,VH__ = torch.linalg.svd(W__)
  Utrunc__ = U__[:,:N]
  Vtrunc__ = VH__[:N,:]
  Strunc__ = torch.diag(S_[:N]).to(Vtrunc__.dtype)
  return Utrunc__, Strunc__ @ Vtrunc__

def is_unitary(A,atol=1e-5):
  vals = torch.abs(torch.linalg.eigvals(A))
  flg_unitary = torch.all(torch.abs(vals - 1) <= atol)
  if not flg_unitary:
    print(vals)
  return flg_unitary

def todB(x_):
  x_ = 10*np.log10(x_)
  x_ -= np.max(x_)
  return x_

def H(X__):
  if isinstance(X__,np.ndarray):
    return np.conj(np.swapaxes(X__,-1,-2))
  elif isinstance(X__,torch.Tensor):
    return torch.conj(torch.transpose(X__,-1,-2))
  
def batch_trace(x):
  return x.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

def extract_diagonals(batch_matrices):
  """
  Extracts the diagonal elements from a batch of square matrices.
  
  Parameters:
  batch_matrices (np.ndarray or torch.Tensor): A 3D array or tensor of shape (F, M, M),
                                                where each MxM matrix is a square matrix and
                                                F is the number of such matrices.
  
  Returns:
  np.ndarray or torch.Tensor: A 2D array or tensor of shape (F, M) containing the diagonal
                              elements of each matrix.
  """
  # Check the type and dimensions of the input
  if isinstance(batch_matrices, np.ndarray):
    if batch_matrices.ndim != 3 or batch_matrices.shape[1] != batch_matrices.shape[2]:
      raise ValueError("Input must be a batch of square matrices, i.e., shape (F, M, M).")
    # Extract diagonals for NumPy arrays
    diagonals = np.diagonal(batch_matrices, axis1=1, axis2=2)
  elif isinstance(batch_matrices, torch.Tensor):
    if batch_matrices.dim() != 3 or batch_matrices.size(1) != batch_matrices.size(2):
      raise ValueError("Input must be a batch of square matrices, i.e., shape (F, M, M).")
    # Extract diagonals for PyTorch tensors
    diagonals = torch.diagonal(batch_matrices, dim1=1, dim2=2)
  else:
    raise TypeError("Input must be either a NumPy array or a PyTorch tensor.")

  return diagonals

def load_matrix_simple(A__,lf):
  M,_ = A__.shape
  if isinstance(A__,np.ndarray):
    mod = np
  elif isinstance(A__,torch.Tensor):
    mod = torch
  else:
    raise(ValueError("load_matrix input was neither an np.ndarray or a torch.Tensor"))
  
  if mod == np:
    return A__ + (lf/M)*mod.abs(mod.trace(A__))*mod.eye(M)
  else:
    return A__ + (lf/M)*mod.abs(mod.trace(A__))*mod.eye(M,device=A__.device)