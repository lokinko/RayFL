import math

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


class RecMetrics(object):
    def __init__(self, top_k):
        self._top_k = top_k
        self._subjects = None  # Subjects which we ran evaluation on

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, top_k):
        self._top_k = top_k

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, subjects):
        """
        args:
            subjects: list, [test_users, test_items, test_scores, negative users, negative items, negative scores]
        """
        assert isinstance(subjects, list)
        self.test_users, self.test_items, self.test_scores = subjects[0], subjects[1], subjects[2]
        self.neg_users, self.neg_items, self.neg_scores = subjects[3], subjects[4], subjects[5]

        # the golden set
        test = pd.DataFrame({'user': self.test_users,
                             'test_item': self.test_items,
                             'test_score': self.test_scores})
        # the full set
        all_data = pd.DataFrame({'user': self.neg_users + self.test_users,
                                 'item': self.neg_items + self.test_items,
                                 'score': self.neg_scores + self.test_scores})
        all_data = pd.merge(all_data, test, on=['user'], how='left')
        # rank the items according to the scores for each user
        all_data['rank'] = all_data.groupby('user')['score'].rank(method='first', ascending=False)
        all_data.sort_values(['user', 'rank'], inplace=True)

        top_k = all_data[all_data['rank'] <= self._top_k]
        self.test_in_top_k = top_k[top_k['test_item'] == top_k['item']]  # golden items hit in the top_K items
        self.user_num = all_data['user'].nunique()

    def cal_auc(self):
        """
        Calculate AUC (Area Under the ROC Curve)

        Args:
            subjects: list, [test_users, test_items, test_scores, negative_users, negative_items, negative_scores]

        Returns:
            float: AUC value
        """
        y_true = np.concatenate([np.ones(len(self.test_scores)), np.zeros(len(self.neg_scores))])
        y_scores = np.concatenate([self.test_scores, self.neg_scores])

        auc = roc_auc_score(y_true, y_scores)
        return auc

    def cal_gauc(self):
        """
        Calculate GAUC (Grouped Area Under the ROC Curve)

        Args:
            subjects: list, [test_users, test_items, test_scores, negative_users, negative_items, negative_scores]

        Returns:
            float: GAUC value
        """

        # Group the scores by user
        y_true = np.concatenate([np.ones(len(self.test_scores)), np.zeros(len(self.neg_scores))])
        y_scores = np.concatenate([self.test_scores, self.neg_scores])
        users = np.concatenate([self.test_users, self.neg_users])

        # Calculate GAUC
        gauc = 0.0
        unique_users = np.unique(users)
        for user in unique_users:
            user_indices = np.where(users == user)[0]
            user_true = y_true[user_indices]
            user_scores = y_scores[user_indices]
            gauc += roc_auc_score(user_true, user_scores)

        gauc /= len(unique_users)
        return gauc

    def cal_hit_ratio(self):
        """Hit Ratio @ top_K"""

        return len(self.test_in_top_k) * 1.0 / self.user_num

    def cal_ndcg(self):
        """NDCG @ top_K"""

        ndcg = self.test_in_top_k['rank'].apply(lambda x: math.log(2) / math.log(1 + x))  # the rank starts from 1
        self.test_in_top_k['ndcg'] = ndcg

        return self.test_in_top_k['ndcg'].sum() * 1.0 / self.user_num
