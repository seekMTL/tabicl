from __future__ import annotations

import sys
import random
import itertools
from collections import OrderedDict
from copy import deepcopy
from typing import List, Optional

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OrdinalEncoder,
    StandardScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
)
from sklearn.utils.validation import check_is_fitted

from .sklearn_utils import validate_data


class RecursionLimitManager:
    """Context manager to temporarily set the recursion limit.

    Parameters
    ----------
    limit : int
        The recursion limit to set temporarily.

    Examples
    --------
    >>> with RecursionLimitManager(4000):
    ...     # Perform operations that require a higher recursion limit
    ...     pass
    """

    def __init__(self, limit):
        self.limit = limit
        self.original_limit = None

    def __enter__(self):
        self.original_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(self.limit)
        return self

    def __exit__(self, type, value, traceback):
        sys.setrecursionlimit(self.original_limit)
        return False  # Return False to propagate exceptions


class TransformToNumerical(TransformerMixin, BaseEstimator):
    """Transform non-numerical data in a DataFrame to numerical representations.

    This transformer automatically detects and converts categorical variables, text features,
    and boolean data types into numerical representations suitable for machine learning models.

    Parameters
    ----------
    verbose : bool, default=False
        Whether to print information about column classifications.

    Attributes
    ----------
    tfm_ : ColumnTransformer or FunctionTransformer
        The fitted transformer that handles the conversion of different column types.

        - If input is a DataFrame: a ``ColumnTransformer`` with ``OrdinalEncoder``
          for categorical columns and ``SimpleImputer`` for numeric columns.
        - If input is not a DataFrame: a ``FunctionTransformer`` that passes data
          through unchanged.
    """
    # 此类职责是将 DataFrame 中的非数值数据（分类、文本、布尔）转换为数值表示，因为机器学习模型只能处理数字
    # DataFrame = pandas 的二维表格，带行索引和列名，每列可以不同类型，是数据科学中最常用的数据容器，类似于 Python 里的 Excel

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def fit(self, X, y=None):
        """Configure transformers for different column types in the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data. If a DataFrame, column types are used to determine
            appropriate transformations.

        y : None
            Ignored.

        Returns
        -------
        self : TransformToNumerical
            Returns self.
        """

        # 定义两种转换器
        # 分类变量 → 整数，如["cat","dog","cat"] → [0, 1, 0]
        cat_tfm = OrdinalEncoder(
            dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1
        )
        # 填充缺失值（均值），如[1, NaN, 3] → [1, 2, 3]
        num_tfm = SimpleImputer()

        # 检查是否是 DataFrame，然后定义合适的数据转换器
        if not hasattr(X, "columns"):  # proxy way to check whether X is a dataframe without importing pandas
            # 逻辑：如果不是 DataFrame，必须是纯数值数组，否则报错，如果不报错说明是纯数值数组，则只需用 SimpleImputer
            # no dataframe, so we can't do column-wise transformations. Instead, we check if it's already numeric and if not, raise an error.
            
            # For compatibility with sklearn's tests
            if issparse(X):
                raise TypeError(
                    "Sparse input is not supported by TabICL. "
                    "Convert X to a dense array, e.g. with X.toarray()."
                )
            X_arr = np.asarray(X)
            try:
                X_arr.astype(np.float64) # 尝试转为浮点数，若转换成功说明原数据全是数值，否则存在非数值
            except (ValueError, TypeError) as e:
                # Preserve the original exception type so that, e.g., object arrays
                # holding non-string/non-number elements still raise a TypeError.
                raise type(e)(
                    "NumPy arrays passed to TabICL must be castable to a numeric dtype, "
                    f"but casting to float64 failed with: {e}. "
                    "If your data contains categorical or string columns, pass it as a pandas "
                    "DataFrame instead, so each column can be typed and preprocessed accordingly."
                ) from None
            self.tfm_ = num_tfm  # 只用 SimpleImputer，只需要填充缺失值

        else:
            # DataFrame 输入处理（核心）

            # 找出分类列（字符串、对象、类别、布尔）
            cat_cols = make_column_selector(dtype_include=["string", "object", "category", "boolean"])(X)
            cat_pos = [X.columns.get_loc(col) for col in cat_cols]

            # 警告高基数分类列（遍历所有分类列，如果某一列的不同值超过 40 个，就标记为"高基数列"）
            # 比如：性别列，只有2种情况，是低基数，直接用 0/1 编码；城市列可能10种，低基数，OrdinalEncoder 可以处理；而用户ID列，高基数，用简单编码会很稀疏，效果差
            high_cardinality_cols = [col for col in cat_cols if X[col].nunique() > 40]
            if high_cardinality_cols:
                import warnings

                warnings.warn(
                    f"The following categorical columns have a cardinality above 40: {high_cardinality_cols}. "
                    "High-cardinality columns might benefit from a better encoding than ordinal encoding, "
                    "e.g. Skrub's TableVectorizer for strings."
                )

            # 找出数值列
            numeric_cols = make_column_selector(dtype_include="number")(X)
            numeric_pos = [X.columns.get_loc(col) for col in numeric_cols]

            # 创建列转换器
            self.tfm_ = ColumnTransformer(
                transformers=[("categorical", cat_tfm, cat_pos), ("continuous", num_tfm, numeric_pos)]
            ) # 分类列用 OrdinalEncoder、数值列用 SimpleImputer

        # 执行拟合
        # 作用：学习转换参数(OrdinalEncoder：学习所有唯一值到整数的映射，SimpleImputer：计算均值)
        self.tfm_.fit(X)

        # 打印调试信息
        if self.verbose and hasattr(self.tfm_, "transformers_"):
            selected_cols = []
            for name, tfm, pos in self.tfm_.transformers_:
                if tfm != "drop":
                    cols = list(X.columns[pos])
                    selected_cols.extend(cols)
                    print(f"Columns classified as {name}: {cols}")

            dropped_cols = set(X.columns).difference(set(selected_cols))
            if len(dropped_cols) >= 1:
                print(f"The following columns are not used due to their data type: {list(dropped_cols)}")

        return self

    def transform(self, X):
        # fit 学转换规则（哪些是分类列？哪些是数值列？映射关系是什么？），transform 用学到的规则执行转换
        """Transform features using the fitted transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features)
            Transformed array with numerical representations.
        """
        return self.tfm_.transform(X)


class UniqueFeatureFilter(TransformerMixin, BaseEstimator):
    """Filter that removes features with only one unique value in the training set.

    Parameters
    ----------
    threshold : int, default=1
        Features with unique values less than or equal to this threshold will be removed.

    Attributes
    ----------
    n_features_in_ : int
        Number of features in the training data.

    n_features_out_ : int
        Number of features after filtering.

    features_to_keep_ : ndarray
        Boolean mask for features to keep.

    Notes
    -----
    1. Features with unique values <= ``threshold`` are removed.
    2. When the input dataset has very few samples
       (:math:`n_{\\text{samples}} \\le \\text{threshold}`), all features are preserved
       regardless of their unique value counts. This is a safety mechanism because:

       - With few samples, it's difficult to reliably assess feature variability.
       - A feature might appear constant in few samples but vary in the complete dataset.
    """

    def __init__(self, threshold: int = 1):
        self.threshold = threshold

    def fit(self, X, y=None):
        """Learn which features to keep based on unique value counts.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Returns self.
        """
        X = validate_data(self, X) # 验证输入数据格式，也设置了self.n_features_in_ = X.shape[1]，即特征数量

        # If there are very few samples, keep all features
        if X.shape[0] <= self.threshold:
            self.features_to_keep_ = np.ones(self.n_features_in_, dtype=bool)
        else:
            # For each feature, check if it has more than threshold unique values
            self.features_to_keep_ = np.array(
                [len(np.unique(X[:, i])) > self.threshold for i in range(self.n_features_in_)]
            )

        self.n_features_out_ = np.sum(self.features_to_keep_)

        return self

    def transform(self, X):
        """Filter features according to unique value counts.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features_out_)
            Transformed array with selected features.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        return X[:, self.features_to_keep_]


class OutlierRemover(TransformerMixin, BaseEstimator):
    """Transformer that clips extreme values based on training data distribution.

    This implementation uses a two-stage Z-score based approach to identify and
    clip outliers:

    1. First stage: Identify values with :math:`|z| > \text{threshold}` standard
       deviations and mark as missing.
    2. Second stage: Recompute statistics without outliers for more robust bounds.
    3. Final stage: Apply log-based clipping to maintain data distribution.

    Parameters
    ----------
    threshold : float, default=4.0
        Values beyond this number of standard deviations are considered outliers,
        i.e., values with :math:`|z| > \text{threshold}`.

    Attributes
    ----------
    n_features_in_ : int
        Number of features in the training data.

    means_ : ndarray of shape (n_features_in_,)
        Mean values per feature after removing outliers.

    stds_ : ndarray of shape (n_features_in_,)
        Standard deviation values per feature after removing outliers.

    lower_bounds_ : ndarray of shape (n_features_in_,)
        Lower bounds for clipping,
        :math:`\\mu - \\text{threshold} \\cdot \\sigma`.

    upper_bounds_ : ndarray of shape (n_features_in_,)
        Upper bounds for clipping,
        :math:`\\mu + \\text{threshold} \\cdot \\sigma`.
    """

    # OutlierRemover 使用两阶段 Z-score 方法来检测和裁剪异常值，并在 transform 时采用软裁剪（对数平滑）而非硬截断

    def __init__(self, threshold: float = 4.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        """Learn clipping bounds from training data using two-stage Z-score method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.

        y : None
            Ignored.

        Returns
        -------
        self : OutlierRemover
            Returns self.
        """

        # 一阶段（只用第一轮统计量）的问题：
        # 第一轮：含异常值的均值可能被拉偏，某个异常值很大把均值拉高，，边界也跟着偏移，有些真正的异常值可能因为边界被拉高而漏掉
        # 两阶段：
        # 第一轮：粗略标记明显的异常值
        # 第二轮：在排除异常值后重新计算，得到真正代表"正常数据"的统计量，边界更准确，对异常值的识别更可靠

        X = validate_data(self, X) # 确保是 2D 数组；设置self.n_features_in_=特征数量

        # First stage: Identify outliers using initial statistics
        self.means_ = np.nanmean(X, axis=0) # 沿axis=0（跨所有样本）计算每列的均值，忽略NaN
        # ddof=1（Delta Degrees of Freedom）：使用样本标准差（分母 n-1）而非总体标准差（分母 n），这是统计学的无偏估计，但若只1个样本，ddof=0，因为n-1=0会除零错误
        self.stds_ = np.nanstd(X, axis=0, ddof=1 if X.shape[0] > 1 else 0) # 计算每列标准差

        # Ensure standard deviations are not zero
        self.stds_ = np.maximum(self.stds_, 1e-6) # 防止标准差为零（当某列所有值相同时）

        # Create a clean copy with outliers replaced by NaN
        X_clean = X.copy() # 深拷贝一份数据，不直接修改原始 X
        # 计算第一轮边界，两变量的 shape (n_features,)
        lower_bounds = self.means_ - self.threshold * self.stds_
        upper_bounds = self.means_ + self.threshold * self.stds_

        # Create masks for values outside bounds
        lower_mask = X < lower_bounds[np.newaxis, :] # 布尔矩阵，True 表该值 < 下界（过低异常）；np.newaxis等同None，在数组指定位置插入一个大小为 1 的新维度
        upper_mask = X > upper_bounds[np.newaxis, :] # 过高异常
        outlier_mask = np.logical_or(lower_mask, upper_mask) # np.logical_or逐元素的逻辑或运算, 合并两个方向的异常值检测，outlier_mask：True 表该位置是异常值（过高或过低）

        # Set outliers to NaN
        X_clean[outlier_mask] = np.nan # 将所有异常值替换为 NaN，这样第二轮计算统计量时，这些异常值就被排除了

        # Second stage: Recompute statistics without outliers
        # 重新计算均值和标准差，自动忽略 NaN 值
        self.means_ = np.nanmean(X_clean, axis=0)
        self.stds_ = np.nanstd(X_clean, axis=0, ddof=1 if X.shape[0] > 1 else 0)

        # Ensure standard deviations are not zero
        self.stds_ = np.maximum(self.stds_, 1e-6)

        # Compute final bounds
        self.lower_bounds_ = self.means_ - self.threshold * self.stds_
        self.upper_bounds_ = self.means_ + self.threshold * self.stds_

        return self

    def transform(self, X):
        """Clip values based on learned bounds with log-based adjustments.

        Values are clipped using soft bounds:

        .. math::

            x_{\\text{clipped}} = \\max\\bigl(-\\log(1+|x|) + L,\\; x\\bigr)

            x_{\\text{clipped}} = \\min\\bigl(\\log(1+|x|) + U,\\; x\\bigr)

        where :math:`L` and :math:`U` are the lower and upper bounds.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features)
            Transformed array with clipped values.
        """
        check_is_fitted(self) # 确保 lower_bounds_ 等属性存在，即已fit
        X = validate_data(self, X, reset=False) # 验证输入格式和特征数匹配
        # 理解：越大的异常值被压缩得越狠，但非一刀切。是在原始值和基于边界的对数惩罚值之间取较优者（maximum 取上限、minimum 取下限），实现一种软裁剪——不是暴力截断，而是用对数函数平滑地拉回来
        X = np.maximum(-np.log1p(np.abs(X)) + self.lower_bounds_, X) # -log(1+|x|) + L
        X = np.minimum(np.log1p(np.abs(X)) + self.upper_bounds_, X) # log(1+|x|) + U

        return X


class CustomStandardScaler(TransformerMixin, BaseEstimator):
    """Custom implementation of standard scaling with clipping.

    Computes the z-score :math:`z = (x - \\mu) / (\\sigma + \\epsilon)` and clips
    the result to ``[clip_min, clip_max]``.

    Parameters
    ----------
    clip_min : float, default=-100
        Lower bound for clipping transformed values.

    clip_max : float, default=100
        Upper bound for clipping transformed values.

    epsilon : float, default=1e-6
        Small constant :math:`\\epsilon` added to the standard deviation to avoid
        division by zero.

    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
        The mean value for each feature in the training set.

    scale_ : ndarray of shape (n_features,)
        The standard deviation for each feature in the training set with
        :math:`\\epsilon` added.
    """

    def __init__(self, clip_min: float = -100, clip_max: float = 100, epsilon: float = 1e-6):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.epsilon = epsilon

    def fit(self, X, y=None):
        """Compute the mean and std to be used for scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation.

        y : None
            Ignored.

        Returns
        -------
        self : CustomStandardScaler
            Returns self.
        """

        # 要求输入X必须是2D数组，若用户只传入一个特征的一维数组（比如单独一列数值），scikit-learn 会报错
        if len(X.shape) == 1:
            # If X is a 1D array, reshape it to 2D
            X = X.reshape(-1, 1) # -1：让 NumPy 自动推断这个维度的大小；第二个维度固定为 1（即 1 列）。即(n,)转为(n, 1)

        X = validate_data(self, X)

        self.mean_ = np.mean(X, axis=0) # 沿着第0轴（即行方向，跨所有样本）计算每个特征的均值
        self.scale_ = np.std(X, axis=0) + self.epsilon # 加上epsilon防止除零

        return self

    def transform(self, X):
        """Standardize features by removing the mean and scaling to unit variance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features)
            Transformed array after scaling and clipping.
        """
        #  将原始数据转为 Z-score 标准化数据，然后裁剪到 [clip_min, clip_max] 范围。

        # 1D 向量检测与重塑
        if len(X.shape) == 1:
            is_vector = True # 表示输入X是 1D 向量
            X = X.reshape(-1, 1)
        else:
            is_vector = False

        check_is_fitted(self) # 检查self.mean_和self.scale_是否存在，确保实例已经过fit()。否则直接调 transform，会抛出 NotFittedError
        X = validate_data(self, X, reset=False) # 验证X是numpy数组格式；reset=False不重新计算 n_features_in_（因为那是 fit 阶段做的事），会检查特征数量是否与self.n_features_in_一致

        X_scaled = (X - self.mean_) / self.scale_ # Z-score 标准化，利用 NumPy 的广播机制执行向量化计算
        # 将缩放后的值限制在默认[-100, 100]，因为极端异常值经过 Z-score 后可能产生极大的绝对值（如 ±1000），可能导致后续归一化（PowerTransformer、QuantileTransformer）不稳定或数值溢出
        X_clipped = np.clip(X_scaled, self.clip_min, self.clip_max)

        return X_clipped.reshape(-1) if is_vector else X_clipped

    def inverse_transform(self, X):
        """Scale back the data to the original representation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to inverse transform.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features)
            Transformed array in original scale.
        """
        # 将标准化后的数据还原到原始尺度。是 transform 的严格数学逆，但非精确可逆，因为被transform裁剪掉的值无法恢复

        if len(X.shape) == 1:
            is_vector = True
            X = X.reshape(-1, 1)
        else:
            is_vector = False

        check_is_fitted(self) # 确保已 fit
        X = validate_data(self, X, reset=False) # 验证输入格式和特征数
        X_out = X * self.scale_ + self.mean_ # 逆向计算

        return X_out.reshape(-1) if is_vector else X_out


class RTDLQuantileTransformer(BaseEstimator, TransformerMixin):
    """Quantile transformer adapted for tabular deep learning models.

    This implementation is based on research from the RTDL group and adds noise to training
    data before applying quantile transformation, improving robustness and generalization.
    It also dynamically adjusts the number of quantiles based on data size as
    :math:`\\min(n_{\\text{samples}} / 30,\\; \\text{n\\_quantiles})` with a minimum of 10.

    Parameters
    ----------
    noise : float, default=1e-3
        Magnitude of Gaussian noise to add relative to feature standard deviations.
        Set to 0 to disable noise addition.

    n_quantiles : int, default=1000
        Maximum number of quantiles to use. The actual number used is dynamically
        determined as :math:`\\min(\\lfloor n / 30 \\rfloor, \\text{n\\_quantiles})`
        with a minimum of 10.

    subsample : int, default=1_000_000_000
        Maximum number of samples used to estimate the quantiles for computational
        efficiency.

    output_distribution : {'uniform', 'normal'}, default='normal'
        Marginal distribution for the transformed data.

    random_state : int or None, default=None
        Seed for random number generation for reproducible noise and quantile sampling.

    Attributes
    ----------
    normalizer_ : QuantileTransformer
        Fitted transformer used to transform the data.

    Notes
    -----
    Adapted from https://github.com/yandex-research/tabular-dl-tabr/blob/75105013189c76bc4f247633c2fb856bc948e579/lib/data.py#L262
    following https://github.com/dholzmueller/pytabkit/blob/949bf81e3964f65a33dd2c252c3713c239c17b2d/pytabkit/models/utils.py#L431
    """

    def __init__(
        self,
        noise: float = 1e-3,
        n_quantiles: int = 1000,
        subsample: int = 1_000_000_000,
        output_distribution: str = "normal",
        random_state: Optional[int] = None,
    ):
        self.noise = noise
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self.output_distribution = output_distribution
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the quantile transformer to training data with optional noise addition.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data to fit the transformer.

        y : None
            Ignored.

        Returns
        -------
        self : RTDLQuantileTransformer
            Returns self.
        """
        # Calculate the number of quantiles based on data size
        n_quantiles = max(min(X.shape[0] // 30, self.n_quantiles), 10)

        # Initialize QuantileTransformer
        normalizer = QuantileTransformer(
            output_distribution=self.output_distribution,
            n_quantiles=n_quantiles,
            subsample=self.subsample,
            random_state=self.random_state,
        )

        # Add noise if required
        X_modified = self._add_noise(X) if self.noise > 0 else X

        # Fit the normalizer
        normalizer.fit(X_modified)

        # Show that it's fitted
        self.normalizer_ = normalizer

        return self

    def transform(self, X, y=None):
        """Transform data using the fitted quantile transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to be transformed.

        y : None
            Ignored.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            The transformed data with distribution specified by
            ``output_distribution``.
        """
        check_is_fitted(self)
        return self.normalizer_.transform(X)

    def _add_noise(self, X):
        """Add noise to the input data proportional to feature standard deviations.

        The noise magnitude is controlled by the 'noise' parameter and is scaled
        inversely to the standard deviation of each feature to ensure
        consistent noise levels across features of different scales.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to add noise to.

        Returns
        -------
        X_noisy : ndarray of shape (n_samples, n_features)
            The input data with added Gaussian noise.
        """
        stds = np.std(X, axis=0, keepdims=True)
        noise_std = self.noise / np.maximum(stds, self.noise)
        rng = np.random.default_rng(self.random_state)
        X_noisy = X + noise_std * rng.standard_normal(X.shape)
        return X_noisy


class PreprocessingPipeline(TransformerMixin, BaseEstimator):
    """Preprocessing pipeline for tabular data.

    This pipeline combines scaling, normalization, and outlier handling.

    Parameters
    ----------
    normalization_method : str, default='power'
        Method for normalization: ``'power'``, ``'quantile'``,
        ``'quantile_rtdl'``, ``'robust'``, ``'none'``.

    outlier_threshold : float, default=4.0
        Z-score threshold for outlier detection. Values with
        :math:`|z| > \text{threshold}` are considered outliers.

    random_state : int or None, default=None
        Random seed for reproducible normalization.

    Attributes
    ----------
    n_features_in_ : int
        Number of features in the training data.

    standard_scaler_ : CustomStandardScaler
        The fitted standard scaler.

    normalizer_ : sklearn transformer or None
        The fitted normalization transformer (``PowerTransformer``,
        ``QuantileTransformer``, ``RTDLQuantileTransformer``, or
        ``RobustScaler``). ``None`` when ``normalization_method='none'``.

    outlier_remover_ : OutlierRemover
        The fitted outlier remover.

    X_transformed_ : ndarray of shape (n_samples, n_features)
        The transformed training input data. Saved for later use to avoid
        recomputation.
    """

    # PreprocessingPipeline 是一个将数值表格数据依次经过
    # Z-score 标准化 → 分布归一化（可选 Power/Quantile/Robust）→ 对数软裁剪异常值 的三阶段预处理流水线，
    # 目的是将任意分布的原始特征转换为均值约 0、接近正态、无极端值且数值稳定的表示，供下游 TabICL 模型直接使用

    def __init__(
        self, normalization_method: str = "power", outlier_threshold: float = 4.0, random_state: Optional[int] = None
    ):
        self.normalization_method = normalization_method
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the preprocessing pipeline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : None
            Ignored.

        Returns
        -------
        self : PreprocessingPipeline
            Returns self.
        """
        X = validate_data(self, X)

        # 1. Apply standard scaling
        self.standard_scaler_ = CustomStandardScaler()
        X_scaled = self.standard_scaler_.fit_transform(X) # Z-score标准缩放

        # 2. Apply normalization
        # 此归一化阶段为了输出分布更接近正态的数据 X_normalized
        if self.normalization_method != "none":
            if self.normalization_method == "power":
                # Yeo-Johnson 变换：使数据分布更接近正态分布（高斯分布），能处理正值和负值；standardize=True：变换后再做一次标准缩放
                self.normalizer_ = PowerTransformer(method="yeo-johnson", standardize=True)
            elif self.normalization_method == "quantile":
                # 将数据映射到正态分布，基于分位数（排序后按位置映射，对异常值更鲁棒）
                self.normalizer_ = QuantileTransformer(output_distribution="normal", random_state=self.random_state)
            elif self.normalization_method == "quantile_rtdl":
                self.normalizer_ = Pipeline(
                    [
                        (
                            "quantile_rtdl",
                            RTDLQuantileTransformer(output_distribution="normal", random_state=self.random_state),
                        ),
                        ("std", StandardScaler()),
                    ]
                )
            elif self.normalization_method == "robust":
                self.normalizer_ = RobustScaler(unit_variance=True)
            else:
                raise ValueError(f"Unknown normalization method: {self.normalization_method}")

            # 保存训练数据Z-score标准化后的最值，shape (1, n_features)。用于transform阶段，若测试数据有训练时未出现的极端值导致归一化失败，用这个范围裁剪回退
            self.X_min_ = np.min(X_scaled, axis=0, keepdims=True)
            self.X_max_ = np.max(X_scaled, axis=0, keepdims=True)
            X_normalized = self.normalizer_.fit_transform(X_scaled) # 执行归一化
        else:
            self.normalizer_ = None
            X_normalized = X_scaled

        # 3. Handle outliers
        self.outlier_remover_ = OutlierRemover(threshold=self.outlier_threshold)
        self.X_transformed_ = self.outlier_remover_.fit_transform(X_normalized) # 异常值处理 软裁剪

        return self

    def transform(self, X):
        """Apply the preprocessing pipeline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X_out : ndarray
            Preprocessed data.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, copy=True)
        # Standard scaling
        # 标准缩放，然后默认裁剪到 [-100, 100]
        X = self.standard_scaler_.transform(X) # 关键：使用的是训练集的均值和标准差，不是测试集的！这确保了训练集和测试集经过相同的尺度变换
        # Normalization 归一化（含异常保护）
        if self.normalizer_ is not None:
            try:
                # this can fail in rare cases if there is an outlier in X that was not present in fit()
                X = self.normalizer_.transform(X) # 使用 fit 阶段学到的归一化参数来转换测试数据
            except ValueError:
                # 异常回退机制
                # clip values to train min/max
                X = np.clip(X, self.X_min_, self.X_max_) # 把超出训练集范围的测试值硬截断到训练集的 min/max 范围内
                X = self.normalizer_.transform(X) # 裁剪后重新归一化
        # Outlier removal
        X = self.outlier_remover_.transform(X) # 异常值软裁剪

        return X


class Shuffler:
    """Utility that generates permutations for ensemble creation.

    This class provides methods to create different types of permutations
    that can be used when creating ensemble variants of datasets.

    Parameters
    ----------
    n_elements : int
        Number of elements to shuffle.

    method : str, default='latin'
        Method used for shuffling:
        - ``'none'``: No shuffling.
        - ``'random'``: Random permutation.
        - ``'latin'``: Latin square permutation.
        - ``'shift'``: Circular shift of elements.

    max_elements_for_latin : int, default=4000
        Maximum number of elements for which Latin square permutations are
        generated. If the number of elements exceeds this limit, random
        permutations are used instead.

    random_state : int or None, default=None
        Random seed for reproducible shuffling.
    """

    # Shuffler 是一个纯粹的索引排列生成器，不接触任何数据，只根据指定的策略（默认Latin Square）生成一组特征或标签的重排索引模式，供 EnsembleGenerator 在创建集成变体时用来打乱列顺序或类别标签
    # 主要输入：n_elements（有多少个元素需要打乱）、method（打乱策略），产出: List[np.ndarray]一个整数索引排列的列表
    # 简言之，Shuffler 不处理原始输入数据 X，不关心你的数据长什么样，只是输出一堆索引

    def __init__(
        self,
        n_elements: int,
        method: str = "latin",
        max_elements_for_latin: int = 4000,
        random_state: Optional[int] = None,
    ):
        self.n_elements = n_elements
        self.method = method
        self.max_elements_for_latin = max_elements_for_latin
        self.random_state = random_state

    def shuffle(self, n_estimators: int) -> List[np.ndarray]:
        """Generate shuffling patterns for ensemble diversity.

        Creates permutations of indices according to the specified method,
        which can be used to reorder elements when creating ensemble variants.

        Parameters
        ----------
        n_estimators : int
            Number of permutations to generate.

            - For ``'none'`` method: Always returns a single pattern with no shuffling.
            - For ``'shift'`` method: Generates all possible circular shifts.
            - For ``'latin'`` method: Generates Latin square permutations.
            - For ``'random'`` method: For small element sets
              (:math:`n_{\\text{elements}} \\le 5`), samples from all possible
              permutations; otherwise generates random permutations.

        Returns
        -------
        list of ndarray
            List of permutation arrays, where each array contains
            indices that can be used to shuffle elements.
        """

        self.rng_ = random.Random(self.random_state)
        indices = list(range(self.n_elements))

        # 默认的打乱方法是Latin，但需要处理大元素集的情况
        # Use the random method if n_elements exceeds the limit for Latin square
        if self.n_elements > self.max_elements_for_latin and self.method == "latin":
            method = "random" # Latin Square 对大元素集计算复杂度太高，所以换成random方式
        else:
            method = self.method

        # No shuffling 无打乱情况
        if method == "none" or n_estimators == 1:
            shuffle_patterns = [indices]
            return shuffle_patterns

        # Generate permutations based on method
        if method == "shift": # 生成所有可能的循环移位
            # All possible circular shifts
            shuffle_patterns = [indices[-i:] + indices[:-i] for i in range(self.n_elements)]
        elif method == "random":
            # Random permutations
            if self.n_elements <= 5:
                # 小元素集：枚举所有排列，然后采样，保证随机排列不重复.总之当排列空间小时，穷举成本很低，但能保证每个集成成员用不同的配置
                all_perms = [list(perm) for perm in itertools.permutations(indices)] # permutation生成所有排列
                shuffle_patterns = self.rng_.sample(all_perms, min(n_estimators, len(all_perms)))
            else:
                # 大元素集：直接生成随机排列，因为无法像小元素那样穷举，但相反其随机排列的可能更多，能取到重复的排列几乎为0
                shuffle_patterns = [self.rng_.sample(indices, self.n_elements) for _ in range(n_estimators)]
        elif method == "latin":
            # Latin square permutations
            with RecursionLimitManager(100000):  # Set a higher recursion limit to avoid recursion error
                shuffle_patterns = self._latin_squares() # Latin Square 特点：每行每列元素不重复
        else:
            raise ValueError(f"Unknown method: {method}. Use 'shift', 'random', 'latin', or 'none'.")

        return shuffle_patterns

    def _latin_squares(self):
        """Generate Latin squares for shuffling.

        Returns
        -------
        list
            List of permutations forming a Latin square.
        """

        # Latin Square 是一个 n×n 的矩阵，其中每行每列都包含 1 到 n 的每个元素恰好一次

        def _shuffle_transpose_shuffle(matrix):
            square = deepcopy(matrix) # 深拷贝，不修改原矩阵
            self.rng_.shuffle(square) # 随机打乱行
            trans = list(zip(*square)) # 转置（行变列）
            self.rng_.shuffle(trans) # 随机打乱转置后的行
            return trans

        def _rls(symbols):
            n = len(symbols)
            if n == 1:
                return [symbols] # 基础情况：只有一个符号
            else:
                sym = self.rng_.choice(symbols) # 随机选择一个符号
                symbols.remove(sym) # 从列表中移除
                square = _rls(symbols) # 递归生成剩余符号的 Latin Square
                square.append(square[0].copy()) # 复制第一行作为新行
                for i in range(n):
                    square[i].insert(i, sym) # 在第 i 行的第 i 个位置插入 sym
                return square

        symbols = list(range(self.n_elements))
        square = _rls(symbols) # 递归生成 Latin Square
        shuffles = _shuffle_transpose_shuffle(square) # 随机变换

        return [list(shuffle) for shuffle in shuffles] # 转换为列表


class EnsembleGenerator(TransformerMixin, BaseEstimator):
    """Generate ensemble variants for robust tabular prediction with TabICL.

    This class creates diverse data variants through:

    1. Applying different normalization techniques.
    2. Permuting feature orders to exploit position-invariance in transformer
       architectures.
    3. For classification: Shuffling class labels to prevent overfitting to
       specific class index patterns.

    Parameters
    ----------
    classification : bool
        Whether to generate ensembles for classification tasks.

    n_estimators : int
        Number of ensemble variants to generate.

    norm_methods : str or list[str] or None, default=None
        Normalization methods to apply:
        - ``'none'``: No normalization.
        - ``'power'``: Yeo-Johnson power transform.
        - ``'quantile'``: Transform feature distribution to approximately
          Gaussian, using the empirical quantiles.
        - ``'quantile_rtdl'``: Version of the quantile transform used
          typically in papers by the RTDL group.
        - ``'robust'``: Scale using median and quantiles.
        If set to None, ``['none', 'power']`` will be applied.

    feat_shuffle_method : str, default='latin'
        Feature permutation strategy:
        - ``'none'``: No shuffling and preserve original feature order.
        - ``'shift'``: Circular shifting.
        - ``'random'``: Random permutation.
        - ``'latin'``: Latin square patterns.

    class_shuffle_method : str, default='shift'
        Class label permutation strategy for classification tasks
        (``classification=True``):
        - ``'none'``: No shuffling and preserve original class labels.
        - ``'shift'``: Circular shifting.
        - ``'random'``: Random permutation.
        - ``'latin'``: Latin square patterns.

    outlier_threshold : float, default=4.0
        Z-score threshold for outlier detection and clipping. Values with
        :math:`|z| > \text{threshold}` are considered outliers.

    random_state : int or None, default=None
        Seed for reproducible ensemble generation.

    Attributes
    ----------
    n_features_in_ : int
        Number of input features after filtering.

    n_classes_ : int
        Number of unique target classes for classification.

    unique_filter_ : UniqueFeatureFilter
        Filter that removes features with only one unique value.

    preprocessors_ : dict
        Maps normalization methods to fitted preprocessing pipelines.

    ensemble_configs_ : OrderedDict
        Generated ensemble configurations, organized by normalization method.
        Keys are normalization methods and values are lists of
        ``(X_shuffle, y_pattern)`` tuples, where ``y_pattern`` is a class
        shuffle for classification or ``None`` for regression.

    feature_shuffles_ : OrderedDict
        Maps normalization methods to lists of feature index permutations.

    class_shuffles_ : OrderedDict
        Maps normalization methods to lists of class index permutations for
        classification.

    X_ : ndarray
        Training feature data after filtering.

    y_ : ndarray
        Training target values.
    """

    def __init__(
        self,
        classification: bool,
        n_estimators: int,
        norm_methods: str | List[str] | None = None,
        feat_shuffle_method: str = "latin",
        class_shuffle_method: str = "shift",
        outlier_threshold: float = 4.0,
        random_state: Optional[int] = None,
    ):
        self.classification = classification
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method

        assert class_shuffle_method in ["none", "shift", "random", "latin"], "Invalid class shuffle method."
        self.class_shuffle_method = class_shuffle_method

        self.outlier_threshold = outlier_threshold
        self.random_state = random_state

    def fit(self, X, y):
        """Create ensemble configurations and fit preprocessing pipelines.

        This method:

        1. Removes features with only one unique value.
        2. Generates diverse ensemble configurations.
        3. Fits preprocessing pipelines for each normalization method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature data.

        y : array-like of shape (n_samples,)
            Training target values.

        Returns
        -------
        self : EnsembleGenerator
            Fitted generator.
        """
        # 验证输入格式，X 转为 numpy ndarray，确保是 2D 数组，并设置 self.n_features_in_ = 特征数量
        validate_data(self, X, y)

        if self.norm_methods is None:
            self.norm_methods_ = ["none", "power"]
        else:
            if isinstance(self.norm_methods, str):
                self.norm_methods_ = [self.norm_methods]
            else:
                self.norm_methods_ = self.norm_methods

        # Filter unique features
        self.unique_filter_ = UniqueFeatureFilter()
        X = self.unique_filter_.fit_transform(X)

        self.X_ = X
        self.y_ = y

        # override n_features_in_ to account for unique feature filtering
        self.n_features_in_ = X.shape[1]
        if self.classification: # 分类任务
            self.n_classes_ = len(np.unique(y))

        self.rng_ = random.Random(self.random_state)
        self.ensemble_configs_, self.feature_shuffles_, y_patterns = self._generate_ensemble()

        if self.classification:
            self.class_shuffles_ = y_patterns

        # Fit preprocessing pipelines
        self.preprocessors_ = {}
        for norm_method in self.ensemble_configs_:
            if norm_method not in self.preprocessors_:
                preprocessor = PreprocessingPipeline(
                    normalization_method=norm_method,
                    outlier_threshold=self.outlier_threshold,
                    random_state=self.random_state,
                )
                preprocessor.fit(X)
                self.preprocessors_[norm_method] = preprocessor

        return self

    def _generate_ensemble(self):
        """Create diverse ensemble configurations grouped by normalization method.

        Returns
        -------
        ensemble_configs : OrderedDict
            Maps normalization methods to shuffle configs.

        X_shuffle_dict : OrderedDict
            Maps normalization methods to lists of feature shuffle patterns.

        y_pattern_dict : OrderedDict
            Maps normalization methods to lists of class shuffles for
            classification or ``None`` patterns for regression.
        """

        # 生成多样化的集成配置，包括：
        # 1、特征打乱模式（feature shuffles）
        # 2、类别标签打乱模式（class shuffles，仅分类任务）
        # 3、按归一化方法分组

        # Generate feature shuffle patterns
        # 生成特征打乱模式
        feat_shuffler = Shuffler( # 创建特征打乱器
            n_elements=self.n_features_in_, method=self.feat_shuffle_method, random_state=self.random_state
        )
        X_shuffles = feat_shuffler.shuffle(self.n_estimators) # 生成 n_estimators 个打乱模式

        # 生成类别标签打乱模式
        if self.classification:
            # For classification, generate class shuffle patterns
            class_shuffler = Shuffler(
                n_elements=self.n_classes_, method=self.class_shuffle_method, random_state=self.random_state
            )
            y_patterns = class_shuffler.shuffle(self.n_estimators)
        else:
            y_patterns = [None] # 回归任务不需要打乱

        # 与归一化方法组合

        # Create configurations combining feature and target patterns
        shuffle_configs = list(itertools.product(X_shuffles, y_patterns))
        self.rng_.shuffle(shuffle_configs)

        shuffle_norm_configs = list(itertools.product(shuffle_configs, self.norm_methods_)) # 笛卡尔积：所有配置 × 所有归一化方法
        shuffle_norm_configs = shuffle_norm_configs[: self.n_estimators] # 截取前 n_estimators 个

        # 按归一化方法分组

        # Reorganize configs so that those with the same normalization method are grouped together
        used_methods = list(set([config[1] for config in shuffle_norm_configs])) # 获取使用的归一化方法列表

        ensemble_configs = OrderedDict()
        X_shuffle_dict = OrderedDict()
        y_pattern_dict = OrderedDict()

        for method in used_methods:
            # 提取该方法下的所有配置
            shuffle_configs = [config[0] for config in shuffle_norm_configs if config[1] == method]
            # 分离特征打乱和类别打乱
            X_shuffle_dict[method] = [config[0] for config in shuffle_configs] # 特征模式
            y_pattern_dict[method] = [config[1] for config in shuffle_configs] # 类别模式
            ensemble_configs[method] = shuffle_configs # 完整配置

        # n_features_in_ = 3 # 3 个特征  n_estimators = 4 # 4 个基学习器
        # feat_shuffle_method = "latin"   norm_methods_ = ["none", "power"]
        # classification = False # 回归任务
        # 返回值可能示例：
        # ensemble_configs = OrderedDict({
        #     "none": [
        #         ([2,1,0], None),    # 学习器1
        #         ([0,2,1], None),    # 学习器3
        #     ],
        #     "power": [
        #         ([2,1,0], None),    # 学习器2
        #         ([0,2,1], None),    # 学习器4
        #     ]
        # })

        # X_shuffle_dict = OrderedDict({
        #     "none":  [[2,1,0], [0,2,1]],
        #     "power": [[2,1,0], [0,2,1]]
        # })

        # y_pattern_dict = OrderedDict({
        #     "none":  [None, None],
        #     "power": [None, None]
        # })

        return ensemble_configs, X_shuffle_dict, y_pattern_dict

    def transform(self, X=None, mode="both", feature_mask=None):
        """Create ensemble data variants for in-context learning.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or None
            Test input data. Required when mode is ``'both'`` or ``'test'``.
            Can be ``None`` when mode is ``'train'``.

        mode : str, default='both'
            Controls what data is returned:

            - ``'both'``: Combines training and test data. Returns
              ``OrderedDict`` mapping normalization methods to
              ``(X_ensemble[n_variants, n_train+n_test, n_features], y_ensemble[n_variants, n_train])``.
            - ``'train'``: Returns only preprocessed and shuffled training
              data. Returns ``OrderedDict`` mapping normalization methods to
              ``(X_train_ensemble[n_variants, n_train, n_features], y_ensemble[n_variants, n_train])``.
            - ``'test'``: Returns only preprocessed and shuffled test data.
              Returns ``OrderedDict`` mapping normalization methods to
              ``(X_test_ensemble[n_variants, n_test, n_features],)``.

        feature_mask : ndarray of shape (n_original_features,) or None, default=None
            Boolean mask where ``True`` indicates masked (all-NaN) columns in
            the *original* feature space (before ``UniqueFeatureFilter``).  When
            provided, masked columns are dropped from both preprocessed training
            and test data, and feature shuffles are remapped to the reduced
            ``[0, K)`` space.  A transient ``masked_feature_shuffles_``
            attribute is stored for the caller to retrieve the remapped shuffles.

        Returns
        -------
        OrderedDict
            Dictionary mapping normalization methods to data tuples.
        """

        # 根据 fit() 学到的配置，生成多样化的集成数据变体

        check_is_fitted(self, ["ensemble_configs_"])  # 确保已 fit
        assert mode in ("both", "train", "test"), f"Invalid mode: {mode}"

        # 处理特征掩码
        # Remap feature shuffles if a feature mask is provided to drop masked columns
        # feature_mask 不需要用户手动传入，它是在分类器/回归器的 predict 方法中自动计算出来的
        if feature_mask is not None:
            # 示例：原始特征索引: [0, 1, 2, 3, 4]
            # feature_mask: [False, True, False, True, False] → 特征 1,3 全是 NaN，需要掩码
            # filtered_mask (应用 UniqueFeatureFilter 后): [False, True, False, True, False]
            # kept_cols: [True, False, True, False, True]  # 保留 0,2,4

            # 映射: 旧索引→新索引 idx_map = {0: 0, 2: 1, 4: 2}  # 0→0, 2→1, 4→2

            # 原始打乱: [2, 4, 0, 3, 1]
            # 重映射后: [1, 2, 0]  # 只保留存在的索引，替换为新索引

            # Map mask from original feature space to filtered space
            # 1. 映射掩码到过滤后的特征空间
            filtered_mask = feature_mask[self.unique_filter_.features_to_keep_]
            kept_cols = ~filtered_mask # 保留的列（False=保留）
            # Build old-index -> new-index mapping for shuffle remapping
            # 2. 构建旧索引→新索引映射
            idx_map = {}
            new_idx = 0
            for old_idx in range(len(filtered_mask)):
                if kept_cols[old_idx]:
                    idx_map[old_idx] = new_idx
                    new_idx += 1

            # Pre-compute remapped feature shuffles per norm method
            # 3. 重新映射特征打乱模式
            self.masked_feature_shuffles_ = OrderedDict()
            for norm_method, shuffle_configs in self.ensemble_configs_.items():
                remapped = []
                for feat_shuffle, _ in shuffle_configs:
                    remapped.append([idx_map[i] for i in feat_shuffle if i in idx_map])
                self.masked_feature_shuffles_[norm_method] = remapped

        # 三种模式

        if mode == "train":
            y = self.y_
            data = OrderedDict()
            for norm_method, shuffle_configs in self.ensemble_configs_.items():
                # 获取预处理后的训练数据
                X_preprocessed = self.preprocessors_[norm_method].X_transformed_
                # 应用特征掩码
                if feature_mask is not None:
                    X_preprocessed = X_preprocessed[:, kept_cols]
                X_ensemble = []
                y_ensemble = []
                for i, (feat_shuffle, y_pattern) in enumerate(shuffle_configs):
                    # 应用特征打乱
                    if feature_mask is not None:
                        feat_shuffle = self.masked_feature_shuffles_[norm_method][i]
                    X_ensemble.append(X_preprocessed[:, feat_shuffle])
                    # 应用类别打乱（分类任务）
                    if self.classification:
                        y_ensemble.append(np.array(y_pattern)[y.astype(int)])
                    else:
                        y_ensemble.append(y)
                # 堆叠所有变体
                # np.stack 在新维度上把数组摞起来，axis=0 表在最前面插入新维度。 效果是把 N 个 (样本, 特征) 的二维数组，变成一个 (N, 样本, 特征) 的三维数组，每个学习器占一层
                data[norm_method] = (np.stack(X_ensemble, axis=0), np.stack(y_ensemble, axis=0))
            return data

        # mode == "test" or "both" requires X
        assert X is not None, "X is required when mode is 'test' or 'both'"
        X = self.unique_filter_.transform(X)

        # Fill masked columns with 0.0 so sklearn transformers don't choke on NaN
        if feature_mask is not None:
            X = np.array(X, dtype=np.float64)
            X[:, filtered_mask] = 0.0 # 用0.0填充掩码列，避免sklearn转换器因NaN而崩溃

        if mode == "test":
            data = OrderedDict()
            for norm_method, shuffle_configs in self.ensemble_configs_.items():
                # 使用已拟合的预处理器转换测试数据
                X_test_preprocessed = self.preprocessors_[norm_method].transform(X)
                # 应用特征掩码
                if feature_mask is not None:
                    X_test_preprocessed = X_test_preprocessed[:, kept_cols]
                X_ensemble = []
                for i, (feat_shuffle, _) in enumerate(shuffle_configs):
                    # 应用特征打乱
                    if feature_mask is not None:
                        feat_shuffle = self.masked_feature_shuffles_[norm_method][i]
                    X_ensemble.append(X_test_preprocessed[:, feat_shuffle])
                # 注意: 只返回 X，没有 y
                data[norm_method] = (np.stack(X_ensemble, axis=0),)
            return data

        # mode == "both"
        y = self.y_
        data = OrderedDict()
        for norm_method, shuffle_configs in self.ensemble_configs_.items():
            preprocessor = self.preprocessors_[norm_method]
            # 获取训练数据（已预处理）
            X_train_pp = preprocessor.X_transformed_
            # 转换测试数据
            X_test_pp = preprocessor.transform(X)
            # 应用特征掩码
            if feature_mask is not None:
                X_train_pp = X_train_pp[:, kept_cols]
                X_test_pp = X_test_pp[:, kept_cols]
            # 拼接训练和测试数据
            X_variant = np.concatenate([X_train_pp, X_test_pp], axis=0)
            X_ensemble = []
            y_ensemble = []
            for i, (feat_shuffle, y_pattern) in enumerate(shuffle_configs):
                # 应用特征打乱
                if feature_mask is not None:
                    feat_shuffle = self.masked_feature_shuffles_[norm_method][i]
                X_ensemble.append(X_variant[:, feat_shuffle])

                # 应用类别打乱（分类任务）
                if self.classification:
                    # Apply class shuffle for classification
                    y_ensemble.append(np.array(y_pattern)[y.astype(int)])
                else:
                    y_ensemble.append(y)

            # 堆叠所有变体
            data[norm_method] = (np.stack(X_ensemble, axis=0), np.stack(y_ensemble, axis=0))

        return data
