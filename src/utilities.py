from collections import defaultdict
from copy import deepcopy
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.utils.metaestimators import _BaseComposition

def _name_estimators(estimators):
    """Generate names for estimators."""

    names = [type(estimator).__name__.lower() for estimator in estimators]
    namecount = defaultdict(int)
    for _, name in zip(estimators, names):
        namecount[name] += 1

    for k, v in list(namecount.items()):
        if v == 1:
            del namecount[k]

    for i in reversed(range(len(estimators))):
        name = names[i]
        if name in namecount:
            names[i] += "-%d" % namecount[name]
            namecount[name] -= 1

    return list(zip(names, estimators))

def _merge_lists(nested_list, high_level_indices=None):
    if high_level_indices is None:
        high_level_indices = list(range(len(nested_list)))

    out = []
    for idx in high_level_indices:
        out.extend(nested_list[idx])

    return tuple(sorted(out))


def _calc_score(
    selector, X, y, indices, groups=None, feature_groups=None, **fit_params
):
    if feature_groups is None:
        feature_groups = [[i] for i in range(X.shape[1])]

    IDX = _merge_lists(feature_groups, indices)
    if selector.cv:
        scores = cross_val_score(
            selector.est_,
            X[:, IDX],
            y,
            groups=groups,
            cv=selector.cv,
            scoring=selector.scorer,
            n_jobs=1,
            pre_dispatch=selector.pre_dispatch,
            fit_params=fit_params,
        )
    else:
        selector.est_.fit(X[:, IDX], y, **fit_params)
        scores = np.array([selector.scorer(selector.est_, X[:, IDX], y)])
    return indices, scores


def _preprocess(X):
    if X.ndim != 2:
        raise ValueError(f"The input X must be 2D array. Got {X.ndim}")

    if type(X).__name__ == "DataFrame":
        features_names = list(X.columns)
        X_ = X.to_numpy(copy=True)
    else:
        # it is numpy array
        features_names = None
        X_ = X.copy()

    return X_, features_names


def _get_featurenames(subsets_dict, feature_idx, feature_names, n_features):
    if feature_names is None:
        feature_names = [str(i) for i in range(n_features)]

    dict_keys = subsets_dict.keys()
    for key in dict_keys:
        subsets_dict[key]["feature_names"] = tuple(
            feature_names[idx] for idx in subsets_dict[key]["feature_idx"]
        )

    if feature_idx is None:
        feature_idx_names = None
    else:
        feature_idx_names = tuple(feature_names[idx] for idx in feature_idx)

    return subsets_dict, feature_idx_names
  

class _BaseXComposition(_BaseComposition):
    def _set_params(self, attr, named_attr, **params):
        # Ordered parameter replacement
        # 1. root parameter
        if attr in params:
            setattr(self, attr, params.pop(attr))

        # 2. single estimator replacement
        items = getattr(self, named_attr)
        names = []
        if items:
            names, estimators = zip(*items)
            estimators = list(estimators)
        for name in list(params.keys()):
            if "__" not in name and name in names:
                # replace single estimator and re-build the
                # root estimators list
                for i, est_name in enumerate(names):
                    if est_name == name:
                        new_val = params.pop(name)
                        if new_val is None:
                            del estimators[i]
                        else:
                            estimators[i] = new_val
                        break
                # replace the root estimators
                setattr(self, attr, estimators)

        # 3. estimator parameters and other initialisation arguments
        super(_BaseXComposition, self).set_params(**params)
        return self