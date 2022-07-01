# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unsupervised scores based on code covariance and mutual information."""
import gin.tf
import numpy as np
import scipy
import torch
from absl import logging
from disentanglement_lib.evaluation.metrics import utils


@gin.configurable(
    "unsupervised_metrics",
    blacklist=["ground_truth_data", "representation_function", "random_state",
               "artifact_dir"])
def unsupervised_metrics(ground_truth_data,
                         representation_function,
                         random_state,
                         artifact_dir=None,
                         num_train=gin.REQUIRED,
                         batch_size=16):
    """Computes unsupervised scores based on covariance and mutual information.

    Args:
      ground_truth_data: GroundTruthData to be sampled from.
      representation_function: Function that takes observations as input and
        outputs a dim_representation sized representation for each observation.
      random_state: Numpy random state used for randomness.
      artifact_dir: Optional path to directory where artifacts can be saved.
      num_train: Number of points used for training.
      batch_size: Batch size for sampling.

    Returns:
      Dictionary with scores.
    """
    del artifact_dir
    scores = {}
    logging.info("Generating training set.")
    mus_train, _ = utils.generate_batch_factor_code(
        ground_truth_data, representation_function, num_train, random_state,
        batch_size)
    num_codes = mus_train.shape[0]
    cov_mus = np.cov(mus_train)
    assert num_codes == cov_mus.shape[0]

    # Weighted ICA measure
    scores["weighted_ica_measure"] = wica_loss(mus_train.T, num_codes, latent_normalization=True).item()

    # Gaussian total correlation.
    scores["gaussian_total_correlation"] = gaussian_total_correlation(cov_mus)

    # Gaussian Wasserstein correlation.
    scores["gaussian_wasserstein_correlation"] = gaussian_wasserstein_correlation(
        cov_mus)
    scores["gaussian_wasserstein_correlation_norm"] = (
            scores["gaussian_wasserstein_correlation"] / np.sum(np.diag(cov_mus)))

    # Compute average mutual information between different factors.
    mus_discrete = utils.make_discretizer(mus_train)
    mutual_info_matrix = utils.discrete_mutual_info(mus_discrete, mus_discrete)
    np.fill_diagonal(mutual_info_matrix, 0)
    mutual_info_score = np.sum(mutual_info_matrix) / (num_codes ** 2 - num_codes)
    scores["mutual_info_score"] = mutual_info_score
    return scores


def kl_gaussians_numerically_unstable(mean_0, cov_0, mean_1, cov_1, k):
    """Unstable version used for testing gaussian_total_correlation."""
    det_0 = np.linalg.det(cov_0)
    det_1 = np.linalg.det(cov_1)
    inv_1 = np.linalg.inv(cov_1)
    return 0.5 * (
            np.trace(np.matmul(inv_1, cov_0)) + np.dot(mean_1 - mean_0,
                                                       np.dot(inv_1, mean_1 - mean_0))
            - k + np.log(det_1 / det_0))


def gaussian_total_correlation(cov):
    """Computes the total correlation of a Gaussian with covariance matrix cov.

    We use that the total correlation is the KL divergence between the Gaussian
    and the product of its marginals. By design, the means of these two Gaussians
    are zero and the covariance matrix of the second Gaussian is equal to the
    covariance matrix of the first Gaussian with off-diagonal entries set to zero.

    Args:
      cov: Numpy array with covariance matrix.

    Returns:
      Scalar with total correlation.
    """
    return 0.5 * (np.sum(np.log(np.diag(cov))) - np.linalg.slogdet(cov)[1])


def gaussian_wasserstein_correlation(cov):
    """Wasserstein L2 distance between Gaussian and the product of its marginals.

    Args:
      cov: Numpy array with covariance matrix.

    Returns:
      Scalar with score.
    """
    sqrtm = scipy.linalg.sqrtm(cov * np.expand_dims(np.diag(cov), axis=1))
    return 2 * np.trace(cov) - 2 * np.trace(sqrtm)


def random_choice_full(input, n_samples, number_of_gausses):
    from torch import multinomial, ones
    if n_samples * number_of_gausses < input.shape[0]:
        replacement = False
    else:
        replacement = True
    idx = multinomial(ones(input.shape[0]), n_samples * number_of_gausses, replacement=replacement)
    sampled = input[idx].reshape(number_of_gausses, n_samples, -1)
    return torch.mean(sampled, axis=1)


def wica_loss(z, z_dim, number_of_gausses=10, latent_normalization=False):
    from torch.distributions import MultivariateNormal

    z = torch.tensor(z)
    if latent_normalization:
        x = (z - z.mean(dim=1, keepdim=True)) / z.std(dim=1, keepdim=True)
    else:
        x = z
    dim = z_dim if z_dim is not None else x.shape[1]
    scale = (1 / dim)
    sampled_points = random_choice_full(x, dim, number_of_gausses)
    cov_mat = (scale * torch.eye(dim)).repeat(number_of_gausses, 1, 1).type(torch.float64)

    mvn = MultivariateNormal(loc=sampled_points,
                             covariance_matrix=cov_mat)

    weight_vector = torch.exp(mvn.log_prob(x.reshape(-1, 1, dim).double()))

    sum_of_weights = torch.sum(weight_vector, axis=0)
    weight_sum = torch.sum(x * weight_vector.T.reshape(number_of_gausses, -1, 1), axis=1)
    weight_mean = weight_sum / sum_of_weights.reshape(-1, 1)

    xm = x - weight_mean.reshape(number_of_gausses, 1, -1)
    wxm = xm * weight_vector.T.reshape(number_of_gausses, -1, 1)

    wcov = (wxm.permute(0, 2, 1).matmul(xm)) / sum_of_weights.reshape(-1, 1, 1)

    diag = torch.diagonal(wcov ** 2, dim1=1, dim2=2)
    diag_pow_plus = diag.reshape(diag.shape[0], diag.shape[1], -1) + diag.reshape(diag.shape[0], -1, diag.shape[1])

    tmp = (2 * wcov ** 2 / diag_pow_plus)

    triu = torch.triu(tmp, diagonal=1)
    normalize = 2.0 / (dim * (dim - 1))
    cost = torch.sum(normalize * triu) / number_of_gausses
    return cost
