from typing import Sequence

import numpy as np

from supg.selector import RecallSelector
import math


class OriginalJointSelector(RecallSelector):
    def select(self) -> Sequence:
        all_inds = super().select()
        set_ids = self.set_ids[~np.isin(self.set_ids, self.sampled)]
        results = self.pos_sampled
        self.total_sampled = len(self.sampled)
        filter_num = len(set_ids)
        results1 = self.data.filter(set_ids[:filter_num])
        results.extend(results1)
        self.sampled = np.concatenate([self.sampled, set_ids[:filter_num]])
        self.total_sampled += filter_num
        return np.array(results)