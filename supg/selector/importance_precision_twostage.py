from typing import Sequence

import numpy as np
import math

from supg.datasource import DataSource
from supg.sampler import Sampler, ImportanceSampler, SamplingBounds
from supg.selector.base_selector import BaseSelector, ApproxQuery


class ImportancePrecisionTwoStageSelector(BaseSelector):
    def __init__(
        self,
        query: ApproxQuery,
        data: DataSource,
        sampler: ImportanceSampler,
        start_samp=100,
        step_size=100
    ):
        self.query = query
        self.data = data
        self.sampler = sampler
        if not isinstance(sampler, ImportanceSampler):
            raise Exception("Invalid sampler for importance")
        self.start_samp = start_samp
        self.step_size = step_size

    def select(self) -> Sequence:
        data_idxs = self.data.get_ordered_idxs()
        n = len(data_idxs)
        T = 1 + 2 * (self.query.budget - self.start_samp) // self.step_size
        # TODO: weights
        x_probs = self.data.get_y_prob()
        # self.sampler.set_weights(np.repeat(1.,n)/n)
        self.sampler.set_weights(np.sqrt(x_probs))
        # self.sampler.set_weights(x_probs ** 2)
        # self.sampler.set_weights(x_probs)

        x_ranks = np.arange(n)
        x_basep = np.repeat((1./n),n)
        x_weights = self.sampler.weights

        n_sample_1 = self.query.budget // 2
        n_sample_2 = self.query.budget - n_sample_1

        samp_ranks = np.sort(self.sampler.sample(max_idx=n, s=n_sample_1))
        samp_basep = x_basep[samp_ranks]
        samp_weights = x_weights[samp_ranks]
        samp_ids = data_idxs[samp_ranks]
        samp_labels = self.data.lookup(samp_ids)
        samp_masses = samp_basep / samp_weights

        delta = self.query.delta
        bounder = SamplingBounds(delta=delta / T)
        tpr_lb, tpr_ub = bounder.calc_bounds(
            fx = samp_labels*samp_masses,
        )
        # cutoff_ub: n_match/y; tpr_ub * n: n_match
        cutoff_ub = int(math.ceil(tpr_ub * n / self.query.min_precision))
        # print('cutoff ub: {}'.format(cutoff_ub))

        samp2_ranks = np.sort(self.sampler.sample(max_idx=cutoff_ub, s=n_sample_2))
        x_weights = self.sampler.weights
        samp2_basep = x_basep[samp2_ranks]
        samp2_weights = x_weights[samp2_ranks]
        samp2_ids = data_idxs[samp2_ranks]
        samp2_labels = self.data.lookup(samp2_ids)
        samp2_masses = samp2_basep / samp2_weights
        # print("ns2: {}, len(samp2): {}".format(n_sample_2, len(samp2_ids)))

        allowed = [0]
        # for s_idx in range(self.start_samp, n_sample_2, self.step_size):
        start = ((n_sample_2 - self.start_samp) // self.step_size) * self.step_size + self.start_samp
        print("start: {}, n_sample_2: {}, samp_2_size: {}".format(start, n_sample_2, len(samp2_ranks)))
        for s_idx in range(start, self.start_samp - 1, -self.step_size):
            print("herep1", s_idx)
            if s_idx + 1 >= len(samp2_ranks):
                continue
            cur_u_idx = samp2_ranks[s_idx]
            # print("curidx: {}, s_idx: {}".format(cur_u_idx, s_idx))
            cur_x_basep = x_basep[:cur_u_idx+1] / np.sum(x_basep[:cur_u_idx+1])
            cur_x_weights = x_weights[:cur_u_idx+1] / np.sum(x_weights[:cur_u_idx+1])
            cur_x_masses = cur_x_basep / cur_x_weights

            cur_subsample_x_idxs = samp2_ranks[:s_idx+1]

            pos_rank_lb, pos_rank_ub = bounder.calc_bounds(
                fx=samp2_labels[:s_idx+1]*cur_x_masses[cur_subsample_x_idxs],
            )
            prec_lb = pos_rank_lb
            print("prec_lb: {}, prec_ub: {}".format(prec_lb, pos_rank_ub))
            if prec_lb > self.query.min_precision:
                print("herep2",s_idx)
                allowed.append(cur_u_idx)
                break

        set_inds = data_idxs[:allowed[-1]]
        self.precision_set_ids = set_inds
        self.precision_critical_value = data_idxs[allowed[-1]]
        # samp_inds = self.data.filter(np.concatenate([samp_ids, samp2_ids]))
        pos_samp_ids1 = np.array([samp_ids[i] for i in range(len(samp_ids)) if samp_labels[i] == 1])
        pos_samp_ids2 = np.array([samp2_ids[i] for i in range(len(samp2_ids)) if samp2_labels[i] == 1])
        pos_samp_ids = np.concatenate([pos_samp_ids1, pos_samp_ids2])
        self.precision_sampled = np.unique(np.concatenate((samp_ids, samp2_ids)))
        self.precision_pos_sampled = np.unique(pos_samp_ids)
        return np.unique(np.concatenate([set_inds, pos_samp_ids]))
