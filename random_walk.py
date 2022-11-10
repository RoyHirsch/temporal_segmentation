import pickle
from typing import Sequence, Optional, Tuple, Mapping, Callable
import functools
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize
from scipy import sparse
from scipy.special import softmax
from scipy.sparse import csgraph
import numpy as np


def get_affinity_matrix(frames: np.array,
                        simillarity_method: str, 
                        post_process_fn: Callable = lambda x: x):
  """Calculate the dense similarity matrix between the frames' reprentations."""

  if simillarity_method == 'cosine':
    w_mat = cosine_similarity(frames, frames)  
  if simillarity_method == 'euclidean':
    w_mat = euclidean_distances(frames, frames)**2
  elif simillarity_method == 'norm_euclidean':
    frames = normalize(frames, axis=1, norm='l2')
    w_mat = euclidean_distances(frames, frames)**2
  return post_process_fn(w_mat)


def sparsify_affine_matrix(w_mat: np.array,
                           m: int,
                           average_method: str,
                           tau: float = 0.5,
                           post_process_fn: Callable = lambda x: x):
  """A method for sparsifying the affine matrix, to inlude 2 neighbors per node.
  Also includes logic for averaging 'm' edges weights for calculating the sparse
  affine matrix.

  Args:
   w_mat: The dense affine matrix, shape (n_frames, n_frames)
   m: The number of neighbors to aggregate weights (from one-side)
   average_method: The method for averaging the m weight's values
   tau: For Daniel's method
   post_process_fn: Post processing of the weights values.
  Returs:
   sparse_w_mat, same shape as w_mat
  """
  n = len(w_mat)
  rs = [np.array([w_mat[i, j] if j < n else 0.0 for j in range(i + 1, i + m + 1)]) for i in range(n)]
  ls = [np.array([w_mat[i, j] if j >= 0 else 0.0 for j in range(i - m, i)]) for i in range(n)]

  r = np.stack(rs, 1)[:,:-1]
  l = np.stack(ls, 1)[:,1:]

  sparse_w_mat = np.zeros_like(w_mat)
  if average_method == 'mean':
    np.fill_diagonal(sparse_w_mat[:, 1:], r.mean(0))
    np.fill_diagonal(sparse_w_mat[1:, :], l.mean(0))

  if average_method == 'weighted_mean':
    # correction for the first m rows 
    wls = []
    for j in range(1, m):
      wls.append(np.concatenate([np.zeros(m - j), np.arange(1, j + 1) / sum(np.arange(j + 1))]))
    wl_v = np.arange(1, m + 1) / sum(np.arange(m + 1))
    wl = np.concatenate([np.stack(wls), np.tile(wl_v, (len(sparse_w_mat) - m, 1))])

    wr = np.flipud(wl)
    r *= wr.T
    l *= wl.T
    np.fill_diagonal(sparse_w_mat[:, 1:], post_process_fn(r.sum(0)))
    np.fill_diagonal(sparse_w_mat[1:, :], post_process_fn(l.sum(0)))

  if average_method == 'sum':
    np.fill_diagonal(sparse_w_mat[:, 1:], r.sum(0))
    np.fill_diagonal(sparse_w_mat[1:, :], l.sum(0))

  elif average_method == 'min':
    np.fill_diagonal(sparse_w_mat[:, 1:], r.min(0))
    np.fill_diagonal(sparse_w_mat[1:, :], l.min(0))

  elif average_method == 'max':
    np.fill_diagonal(sparse_w_mat[:, 1:], r.max(0))
    np.fill_diagonal(sparse_w_mat[1:, :], l.max(0))

  elif average_method == 'daniel':
    np.fill_diagonal(sparse_w_mat[:, 1:], (r >= tau).sum(0) / m)
    np.fill_diagonal(sparse_w_mat[1:, :], (l >= tau).sum(0) / m)

  return sparse_w_mat


def power(x, a):
  return np.power(x, a)


def solve(laplacian, prior, gamma=1e-2):
  """Solve Ax = b with the for s phases with temporal prior.
   can be a vector or a scalar.
  """

  n = len(laplacian)
  if isinstance(gamma, (int, float)):
    gamma_vec = np.full(shape=(n,), fill_value=gamma) 
  else:
    gamma_vec = gamma_vec

  prior = prior.T
  lap_sparse = sparse.csr_matrix(laplacian) 
  gamma_sparse = sparse.coo_matrix((gamma_vec, (range(n), range(n))))
  A_sparse = lap_sparse + gamma_sparse
  A_sparse = A_sparse.tocsc()
  solver = sparse.linalg.factorized(A_sparse.astype(np.double))
  X = np.array([solver(gamma_vec * label_prior) for label_prior in prior])
  return X


def predict(laplacian, prior, gamma=1e-2):
  """Solve and predict with argmax"""

  X = solve(laplacian, prior, gamma=gamma)
  preds = np.argmax(X, axis=0)
  return preds


def extract_prior(labels, timestamps, actions_dict, value=1., smooth=None):
  # sparse_prior.shape [num_frames, num_classes]

  duration = len(labels)
  sparse_prior = np.zeros((len(labels), len(actions_dict)))
  for j in timestamps:
    i = labels[j]
    sparse_prior[j, i] = value
    if smooth:
      for m, n in enumerate(range(j - smooth, j)):
        if n >= 0:
          sparse_prior[n, i] = (value / (smooth + 1) * (m + 1))
      for m, n in enumerate(range(j + smooth, j, -1)):
        if n < duration:
          sparse_prior[n, i] = (value / (smooth + 1) * (m + 1))
  return sparse_prior


PARAMS = {'seed': 0,
          'topk': 1,
          'n_neighbors': 50,
          'average_method': 'min',
          'affine_power_a': 6,
          'prior_lambda': 5e-3,
          'smooth': 10}


def run_random_walk(frames, prior, params=PARAMS):
  w_mat = get_affinity_matrix(frames,
                              simillarity_method='cosine',
                              post_process_fn=functools.partial(power,
                                                      a=params['affine_power_a']))
  sparse_w_mat = sparsify_affine_matrix(w_mat,
                                        m=params['n_neighbors'],
                                        average_method=params['average_method'])
  lap = csgraph.laplacian(sparse_w_mat,
                          normed=False)

  preds = predict(lap, prior, params['prior_lambda'])
  return preds


def e2e_random_walk(frames, labels, timestamps, actions_dict, params=PARAMS):
    sparse_prior = extract_prior(labels, timestamps, actions_dict, value=1., smooth=params['smooth'])
    return run_random_walk(frames, sparse_prior, params)
