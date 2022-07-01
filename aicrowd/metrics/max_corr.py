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

"""Implementation of the maximal correlation
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin.tf
import numpy as np
from absl import logging
from disentanglement_lib.evaluation.metrics import utils


@gin.configurable(
    "max_corr",
    blacklist=["ground_truth_data", "representation_function", "random_state",
               "artifact_dir"])
def compute_max_corr(ground_truth_data,
                     representation_function,
                     random_state,
                     num_train=100,
                     batch_size=16,
                     artifact_dir=None):
    """Computes the maximal correlation score.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    num_train: Number of points used for training.
    batch_size: Batch size for sampling.

  Returns:
    Dictionary with maximal correlation score.
  """
    del artifact_dir

    logging.info("Generating training set.")
    mus, ys = utils.generate_batch_factor_code(
        ground_truth_data, representation_function, num_train,
        random_state, batch_size)

    logging.info("Computing max_corr score.")
    current_factors, _ = \
        ground_truth_data.sample(num_train, random_state)
    num_codes = mus.shape[0]
    if num_codes < 7:
        max_corr = get_max_corr_perm(current_factors.T, ys)
    else:
        max_corr = get_max_corr_np(current_factors.T, ys)

    score_dict = {"max_corr": max_corr}
    return score_dict


def get_max_corr_perm(x, y):
    from itertools import permutations
    x_centered = x - np.mean(x, 0, keepdims=True)
    y_centered = y - np.mean(y, 0, keepdims=True)
    sd_x = np.sqrt(np.mean(x_centered ** 2, 0, keepdims=True))
    x_norm = x_centered / sd_x
    sd_y = np.sqrt(np.mean(y_centered ** 2, 0, keepdims=True))
    corrs = []
    for x_perm in permutations(x_norm.T):
        x_perm = np.stack(x_perm)
        cov_diag = np.mean(x_perm.T * y_centered, 0) / sd_y
        corrs.append(np.mean(np.abs(cov_diag)))
    return np.max(corrs)


def get_max_corr_np(x, y):
    """This function is only for quick approximate evaluation purposes.

    Use 'get_max_corr_perm' for the actual evaluation.

    """
    corr_xy = get_corr_xy_np(x, y)
    return np.mean(np.max(np.abs(corr_xy), 0))


def get_corr_xy_np(x, y):
    x_centered = x - np.mean(x, 0, keepdims=True)
    y_centered = y - np.mean(y, 0, keepdims=True)
    cov_xy = np.dot(x_centered.T, y_centered) / np.float32(x.shape[0])
    sd_x = np.sqrt(np.mean(x_centered ** 2, 0, keepdims=True))
    sd_y = np.sqrt(np.mean(y_centered ** 2, 0, keepdims=True))
    corr_xy = cov_xy / np.dot(sd_x.T, sd_y)
    return corr_xy
