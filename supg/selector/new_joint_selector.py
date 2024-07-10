# -*- coding: utf-8 -*-
from typing import Sequence

import numpy as np

from supg.selector import RecallSelector, ImportancePrecisionTwoStageSelector
import math


class NewJointSelector(RecallSelector, ImportancePrecisionTwoStageSelector):
    def __init__(self, query, data, sampler, start_samp=10, step_size=10, sample_mode="sqrt", verbose=False):
        query.budget = query.budget // 2
        RecallSelector.__init__(self, query, data, sampler, sample_mode, verbose)
        ImportancePrecisionTwoStageSelector.__init__(self, query, data, sampler, start_samp, step_size)
        # for recall at first, then for recall and precision
        self.sampled = None
        self.pos_sampled = []
        # for recall
        self.set_ids = None
        self.critical_value = 0
        # for precision
        self.precision_sampled = None
        self.precision_pos_sampled = None
        self.precision_set_ids = None
        self.precision_critical_value = 0
        self.total_sampled = 0

    def select(self) -> Sequence:
        _ = RecallSelector.select(self)
        _ = ImportancePrecisionTwoStageSelector.select(self)
        self.sampled = np.unique(np.concatenate((self.sampled, self.precision_sampled)))
        self.total_sampled = len(self.sampled)
        self.pos_sampled = np.unique(np.concatenate((self.pos_sampled, self.precision_pos_sampled)))
        precision_set_ids_notsamp = self.precision_set_ids[~np.isin(self.precision_set_ids, self.sampled)]
        print("precision set size", len(self.precision_set_ids), "recall set size", len(self.set_ids))
        set_ids_middle = self.set_ids[~np.isin(self.set_ids, self.precision_set_ids)]
        # if the precision threshold is even lower than the recall threshold, return the precision set
        if len(set_ids_middle) == 0:
            return np.concatenate((self.pos_sampled, precision_set_ids_notsamp))
        set_ids_middle_pos_sampled = set_ids_middle[np.isin(set_ids_middle, self.pos_sampled)]
        set_ids_middle_notsamp = set_ids_middle[~np.isin(set_ids_middle, self.sampled)]
        # 初始的tp只算了在中间的那块的positive sampled
        tp = len(set_ids_middle_pos_sampled)
        tpfp = len(set_ids_middle_pos_sampled) + len(set_ids_middle_notsamp)
        results = np.concatenate((self.pos_sampled, precision_set_ids_notsamp))
        while tp < tpfp * self.query.min_precision:
            print("herejoint", tp, tpfp, tpfp * self.query.min_precision, self.total_sampled)
            filter_num = math.ceil(tpfp * self.query.min_precision - tp)
            results1 = self.data.filter(set_ids_middle_notsamp[:filter_num])
            self.sampled = np.concatenate([self.sampled, set_ids_middle_notsamp[:filter_num]])
            results = np.concatenate((results, results1))
            tp += len(results1)
            tpfp = tpfp - (filter_num - len(results1))
            self.total_sampled += filter_num
            set_ids_middle_notsamp = set_ids_middle_notsamp[filter_num:]
        results = np.concatenate((results, set_ids_middle_notsamp))
        return results



