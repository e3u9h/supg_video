from typing import Sequence

import numpy as np

from supg.selector import RecallSelector
import math


class JointSelector(RecallSelector):
    def select(self) -> Sequence:
        all_inds = super().select()
        set_ids = self.set_ids
        tp = len(self.pos_sampled)
        tpfp = len(all_inds)
        results = self.pos_sampled
        self.total_sampled = self.query.budget
        while tp < tpfp * self.query.min_precision:
            print("herejoint",tp,tpfp, tpfp*self.query.min_precision, self.total_sampled)
            filter_num = math.ceil(tpfp*self.query.min_precision-tp)
            results1 = self.data.filter(set_ids[:filter_num])
            results.extend(results1)
            tp += len(results1)
            tpfp = tpfp - (filter_num - len(results1))
            self.total_sampled += filter_num
            set_ids = set_ids[filter_num:]
        return np.array(results)
