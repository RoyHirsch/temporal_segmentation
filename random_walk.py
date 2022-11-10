from typing import Optional, Mapping, Callable, Union, Sequence, Any
import functools
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize
from scipy import sparse
from scipy.sparse import csgraph
import numpy as np


def get_affinity_matrix(frames: np.array,
                        simillarity_method: str, 
                        post_process_fn: Callable = lambda x: x):
  """Calculates the dense similarity matrix between the frames' reprentations.
  
  Attributse:
    frames: A numpy array with shape [num_frames, hidden_dim].
    simillarity_method: The simillarity method name, supports: [cosine, euclidean, norm_euclidean].
    post_process_fn: Optional function for post processing.

  Returns:
    The scores matrix with shape [num_frames, num_frames]
  """

  if simillarity_method == 'cosine':
    weights_matrix = cosine_similarity(frames, frames)  
  if simillarity_method == 'euclidean':
    weights_matrix = euclidean_distances(frames, frames) ** 2
  elif simillarity_method == 'norm_euclidean':
    frames = normalize(frames, axis=1, norm='l2')
    weights_matrix = euclidean_distances(frames, frames) ** 2
  return post_process_fn(weights_matrix)


def sparsify_affine_matrix(weights_matrix: np.array,
                           m: int,
                           average_method: str,
                           tau: float = 0.5,
                           post_process_fn: Callable = lambda x: x):
  """Sparsifies the affine matrix.

  Keep only scores of the current frame and nearest neighbors (t-1, t, t+1).
  Also includes a mechnism for averaging the weights of 'm' neighbors (from each side).

  Attributse:
   w_mat: The dense affine matrix with shape [num_frames, num_frames].
   m: The number of neighbors to aggregate weights (from one-side).
   average_method: The method for averaging the m weight's values, supports: [sum, min, mean, weighted_mean, max, daniel].
   tau: For Daniel's method (see Overleaf).
   post_process_fn: Optional function for post processing.

  Returs:
   sparse_w_mat, same shape as w_mat
  """
  n = len(weights_matrix)
  rs = [np.array([weights_matrix[i, j] if j < n else 0.0 for j in range(i + 1, i + m + 1)]) for i in range(n)]
  ls = [np.array([weights_matrix[i, j] if j >= 0 else 0.0 for j in range(i - m, i)]) for i in range(n)]

  r = np.stack(rs, 1)[:,:-1]
  l = np.stack(ls, 1)[:,1:]

  sparse_weights_matrix = np.zeros_like(weights_matrix)
  if average_method == 'mean':
    np.fill_diagonal(sparse_weights_matrix[:, 1:], r.mean(0))
    np.fill_diagonal(sparse_weights_matrix[1:, :], l.mean(0))

  if average_method == 'weighted_mean':
    # correction for the first m rows 
    wls = []
    for j in range(1, m):
      wls.append(np.concatenate([np.zeros(m - j), np.arange(1, j + 1) / sum(np.arange(j + 1))]))
    wl_v = np.arange(1, m + 1) / sum(np.arange(m + 1))
    wl = np.concatenate([np.stack(wls), np.tile(wl_v, (len(sparse_weights_matrix) - m, 1))])

    wr = np.flipud(wl)
    r *= wr.T
    l *= wl.T
    np.fill_diagonal(sparse_weights_matrix[:, 1:], post_process_fn(r.sum(0)))
    np.fill_diagonal(sparse_weights_matrix[1:, :], post_process_fn(l.sum(0)))

  if average_method == 'sum':
    np.fill_diagonal(sparse_weights_matrix[:, 1:], r.sum(0))
    np.fill_diagonal(sparse_weights_matrix[1:, :], l.sum(0))

  elif average_method == 'min':
    np.fill_diagonal(sparse_weights_matrix[:, 1:], r.min(0))
    np.fill_diagonal(sparse_weights_matrix[1:, :], l.min(0))

  elif average_method == 'max':
    np.fill_diagonal(sparse_weights_matrix[:, 1:], r.max(0))
    np.fill_diagonal(sparse_weights_matrix[1:, :], l.max(0))

  elif average_method == 'daniel':
    np.fill_diagonal(sparse_weights_matrix[:, 1:], (r >= tau).sum(0) / m)
    np.fill_diagonal(sparse_weights_matrix[1:, :], (l >= tau).sum(0) / m)

  return sparse_weights_matrix


def power(x, a):
  return np.power(x, a)


def solve(laplacian_matix: np.ndarray,
          prior: np.ndarray,
          gamma:Union[np.ndarray, int] = 1e-2):
  """Solves the array of linear equesions.

  Solves (L + gamma * I) x_s = gamma * z_j, where:
    L is laplacian matrix with shape [num_frames, num_frames]
    z_j is the prior vector with shape [num_frames]
    Gamma is a weight factor that controls the prior strenght, can be a vector or a scalar.
  This function solves this linear equesion for every phase/class.

  Attributse:
    laplacian_matix: The laplacian matrix with shape [num_frames, num_frames].
    prior: The prior matrix with shape [num_phases, num_frames].
    gamma: Weight factor that controls the prior strenght.

  Returns:
    The phase/class probability per frame, a matrix with shape:[num_phases, num_frames]
  """
  n = len(laplacian_matix)
  assert n == prior.shape[1]

  if isinstance(gamma, (int, float)):
    gamma_vector = np.full(shape=(n,), fill_value=gamma) 
  else:
    gamma_vector = gamma_vector

  lap_sparse = sparse.csr_matrix(laplacian_matix) 
  gamma_sparse = sparse.coo_matrix((gamma_vector, (range(n), range(n))))
  A_sparse = lap_sparse + gamma_sparse
  A_sparse = A_sparse.tocsc()
  solver = sparse.linalg.factorized(A_sparse.astype(np.double))
  X = np.array([solver(gamma_vector * label_prior) for label_prior in prior])
  return X


def predict(laplacian_matix: np.ndarray,
            prior: np.ndarray,
            gamma: Union[np.ndarray, int] = 1e-2):
  """Solves and predicts the array of linear equesions.
  
    Solves (L + gamma * I) x_s = gamma * z_j, where:
    L is laplacian matrix with shape [num_frames, num_frames]
    z_j is the prior vector with shape [num_frames]
    Gamma is a weight factor that controls the prior strenght, can be a vector or a scalar.
  This function solves this linear equesion for every phase/class and returns the class prediction per frame.

  Attributse:
    laplacian_matix: The laplacian matrix with shape [num_frames, num_frames].
    prior: The prior matrix with shape [num_phases, num_frames].
    gamma: Weight factor that controls the prior strenght.

  Returns:
    The phase/class prediction per frame, a vector:[num_frames]
"""

  X = solve(laplacian_matix, prior, gamma)
  preds = np.argmax(X, axis=0)
  return preds


def extract_timestamps_based_sparse_prior(labels: np.ndarray,
                                          timestamp_indices: Union[Sequence[int], np.ndarray],
                                          num_phases: int,
                                          value: float = 1.,
                                          smooth: Optional[int] = None):
  """Extracts sparse prior based on timestamps.

  Given a list of timestamps indices, generate a sparse prior matrix with shape [num_phases, num_frames].
  This sparse matrix contains only the timestamps labels and the other entries are masked.
  The functino also with smoothing the timestams labels.

  Attributse:
    labels: A list of ground true lables with shape [num_frames].
    timestamps: A list of timestamps indices.
    num_phases: The number of phases.
    value: The value for the timestamps locations.
    smooth: Optional smoothing factor.

  Returns:
    A sparse_prior matrix with shape [num_phases, num_frames]
  """

  num_frames = len(labels)
  sparse_prior = np.zeros((num_phases, num_frames))
  for j in timestamp_indices:
    i = labels[j]
    sparse_prior[j, i] = value
    if smooth:
      for m, n in enumerate(range(j - smooth, j)):
        if n >= 0:
          sparse_prior[n, i] = (value / (smooth + 1) * (m + 1))
      for m, n in enumerate(range(j + smooth, j, -1)):
        if n < num_frames:
          sparse_prior[n, i] = (value / (smooth + 1) * (m + 1))
  return sparse_prior

DEFAULT_PARAMS = {
          'num_neighbors': 50,
          'average_method': 'min',
          'affine_weights_power_factor': 6,
          'prior_lambda': 5e-3,
          'smooth': 10
          }


def predict_random_walk_with_prior(frames: np.ndarray,
                                   prior: np.ndarray,
                                   params: Mapping[str, Any] = DEFAULT_PARAMS):
  """Runs and predicts the random walk with prior.

  Attributse:
    frames: The video's frames with shape [num_frames, hidden_dim].
    prior: The prior matrix with shape [num_phases, num_frames].
    num_phases: The number of phases.
    params: A dictionary with hyper-parameters.

  Returns:
    The phase/class prediction per frame, a vector:[num_frames]

  """
  weights_matrix = get_affinity_matrix(frames,
                              simillarity_method='cosine',
                              post_process_fn=functools.partial(power,
                                                      a=params['affine_weights_power_factor']))
  sparse_weights_matrix = sparsify_affine_matrix(weights_matrix,
                                        m=params['num_neighbors'],
                                        average_method=params['average_method'])
  laplacian_matrix = csgraph.laplacian(sparse_weights_matrix,
                          normed=False)

  preds = predict(laplacian_matrix, prior, params['prior_lambda'])
  return preds


def predict_random_walk_with_timestamps(frames, labels, timestamps, actions_dict, params=DEFAULT_PARAMS):
  """Runs and predicts the random walk with timestamps prior."""

  sparse_prior = extract_timestamps_based_sparse_prior(labels, timestamps, actions_dict, value=1., smooth=params['smooth'])
  return predict_random_walk_with_prior(frames, sparse_prior, params)


if __name__ == '__main__':
  n = 10
  c = 3
  l = np.random.normal(1,0,(n,n))
  prior = np.random.normal(1,0,(c,n))
  predict(laplacian_matix=l, prior=prior)