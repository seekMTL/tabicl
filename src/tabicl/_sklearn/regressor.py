from __future__ import annotations

import warnings
from pathlib import Path
import multiprocessing as mp
from collections import OrderedDict
from typing import Optional, List, Dict

import numpy as np
import torch

from sklearn.base import RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError

from .base import TabICLBaseEstimator
from .preprocessing import TransformToNumerical, EnsembleGenerator
from .sklearn_utils import validate_data, _num_samples

from tabicl import InferenceConfig
from tabicl._model.tabicl import TabICL
from tabicl._model.kv_cache import TabICLCache


class TabICLRegressor(RegressorMixin, TabICLBaseEstimator):
    """Tabular In-Context Learning (TabICL) Regressor with scikit-learn interface.

    This regressor applies TabICL to tabular data regression, using an ensemble
    of transformed dataset views to improve predictions. The ensemble members are
    created by applying different normalization methods and feature permutations.

    Parameters
    ----------
    n_estimators : int, default=8
        Number of estimators for ensemble predictions.

    norm_methods : str or list[str] or None, default=None
        Normalization methods to apply:
        - 'none': No normalization
        - 'power': Yeo-Johnson power transform
        - 'quantile': Transform features to an approximately normal distribution.
        - 'quantile_rtdl': Quantile transform that adds noise to training data before fitting.
        - 'robust': Scale using median and quantiles
        Can be a single string or a list of methods to use across ensemble members.
        When set to None, it will use ["none", "power"].

    feat_shuffle_method : str, default='latin'
        Feature permutation strategy:
        - 'none': No shuffling and preserve original feature order
        - 'shift': Circular shifting of feature columns
        - 'random': Random permutation of features
        - 'latin': Latin square patterns for systematic feature permutations

    outlier_threshold : float, default=4.0
        Z-score threshold for outlier detection and clipping. Values with
        :math:`|z| > \text{threshold}` are considered outliers.

    batch_size : Optional[int], default=8
        Batch size for inference. If None, all ensemble members are processed in a single batch.
        Adjust this parameter based on available memory. Lower values use less memory but may
        be slower.

    kv_cache : bool or str, default=False
        Controls caching of training data computations to speed up subsequent
        ``predict`` calls. The cache is built during ``fit()``.

        - False: No caching.
        - True or "kv": Cache key-value projections from both column embedding
          and ICL transformer layers. Fast inference but memory-heavy for large
          training sets.
        - "repr": Cache column embedding KV projections and row interaction outputs
          (representations). Uses ~24x less memory than "kv" for the ICL part,
          at the cost of re-running the ICL transformer at predict time.

        The cache retains whatever dtype the model produced during ``fit()``
        (float16 when AMP is active, float32 otherwise). If the cache is later
        loaded on CPU or on CUDA without AMP, the tensors are automatically
        upcast to float32 to avoid dtype-mismatch errors.

    model_path : Optional[str or Path], default=None
        Path to the pre-trained model checkpoint file.

        - If provided and the file exists, it's loaded directly.
        - If provided but the file doesn't exist and `allow_auto_download` is true, the version
          specified by `checkpoint_version` is downloaded from Hugging Face Hub (repo: 'jingang/TabICL')
          to this path.
        - If `None` (default), the version specified by `checkpoint_version` is downloaded from
          Hugging Face Hub (repo: 'jingang/TabICL') and cached locally in the default
          Hugging Face cache directory (typically `~/.cache/huggingface/hub`).

    allow_auto_download : bool, default=True
        Whether to allow automatic download if the pretrained checkpoint cannot be found at the
        specified `model_path`.

    checkpoint_version : str, default='tabicl-regressor-v2-20260212.ckpt'
        Specifies which version of the pre-trained model checkpoint to use when `model_path`
        is `None` or points to a non-existent file (and `allow_auto_download` is true).
        Checkpoints are downloaded from https://huggingface.co/jingang/TabICL.

    device : Optional[str or torch.device], default=None
        Device to use for inference. If None, automatically selects CUDA if
        available, otherwise CPU. Can be specified as a string (``'cuda'``,
        ``'cpu'``, ``'mps'``) or a ``torch.device`` object. MPS (Apple Silicon
        GPU) is supported but must be explicitly requested.

    use_amp : bool or "auto", default="auto"
        Controls automatic mixed precision (AMP) for inference.
        - True / False: force on / off.
        - "auto": Automatically enable AMP based on input data size using the following heuristic:

            +--------------------------------------+-------+-------+
            | Regime                               |  AMP  |  FA3  |
            +======================================+=======+=======+
            | Small  (n < 1024 & feat < 60)        |  off  |  off  |
            +--------------------------------------+-------+-------+
            | Medium (above small, n < 10240)      |  on   |  off  |
            +--------------------------------------+-------+-------+
            | Large  (n >= 10240)                  |  on   |  on   |
            +--------------------------------------+-------+-------+

            The above heuristic is based on the observation that AMP can introduce overhead that outweighs
            its benefits for small inputs. In addition, it assumes that the training set is large relative to
            the test set and does not account for KV-cache scenarios. If it is suboptimal for your workload,
            set it explicitly.

    use_fa3 : bool or "auto", default="auto"
        Whether to use Flash Attention 3 that can speed up inference for large datasets on NVIDIA Hopper
        GPUs like H100. Only effective when FA3 is installed.
        - True / False: force on / off.
        - "auto": Automatically enable FA3 based on input data size using a simple heuristic (see above).

    offload_mode : str or bool, default='auto'
        Controls where column-wise embedding outputs are stored during inference.
        Column-wise embedding produces a large tensor of shape
        (batch_size, n_rows, n_columns, embed_dim) which is the main memory bottleneck.
        Available options:
        - ``'auto'``: Automatically choose based on available memory (default).
        - ``'gpu'`` or ``False``: Keep on GPU. Fastest but limited by VRAM.
        - ``'cpu'`` or ``True``: Offload to CPU memory.
        - ``'disk'``: Offload to memory-mapped files (requires ``disk_offload_dir``).

        It only affects column-wise embedding (COL_CONFIG). For finer-grained control
        over all components, use ``inference_config``.

    disk_offload_dir : Optional[str], default=None
        Directory for memory-mapped files used when ``offload_mode='disk'`` or when
        ``offload_mode='auto'`` falls back to disk offloading.
        It only affects column-wise embedding (COL_CONFIG). For finer-grained control
        over all components, use ``inference_config``.

    random_state : int or None, default=42
        Random seed for reproducibility of ensemble generation, affecting feature
        shuffling and other randomized operations.

    n_jobs : int or None, default=None
        Number of threads to use for PyTorch in case the model is run on CPU.
        None means using the PyTorch default, which is the number of physical CPU cores.
        Negative numbers mean that :math:`\\max(1, n_{\\text{logical\\_cores}} + 1 + \\text{n\\_jobs})`
        threads will be used. In particular, ``n_jobs=-1`` means that all logical cores
        will be used.

    verbose : bool, default=False
        Whether to print detailed information during inference.

    inference_config : Optional[InferenceConfig | Dict[str, Dict[str, Any]]], default=None
        Configuration for inference settings. This parameter provides fine-grained control
        over the three transformers in TabICL (column-wise, row-wise, and in-context learning).

        WARNING: This parameter should only be used by advanced users who understand the internal
        architecture of TabICL and need precise control over inference.

        When None (default):
            - A new InferenceConfig object is created with default settings
            - The ``device``, ``use_amp``, ``use_fa3``, ``offload_mode``, ``disk_offload_dir``, and ``verbose``
              parameters from the class initialization are applied to the relevant components

        When Dict with allowed top-level keys "COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG":
            - A new InferenceConfig object is created with default settings
            - Any values explicitly specified in the dictionary will override default defaults
            - ``device``, ``use_amp``, ``use_fa3``, ``offload_mode``, ``disk_offload_dir``, and ``verbose``
              from the class initialization are used if they are not specified in the dictionary

        When InferenceConfig:
            - The provided InferenceConfig object is used directly without modification
            - ``device``, ``use_amp``, ``use_fa3``, ``offload_mode``, ``disk_offload_dir``, and ``verbose``
              from the class initialization are ignored
            - All settings must be explicitly defined in the provided InferenceConfig object

    Attributes
    ----------
    n_features_in_ : int
        Number of features in the training data.

    n_samples_in_ : int
        Number of samples in the training data.

    feature_names_in_ : ndarray of shape ``(n_features_in_,)`` or None
        Feature names seen during ``fit``. Only set when the input ``X`` has
        feature names (e.g., a pandas DataFrame with string column names).

    X_encoder_ : TransformToNumerical
        Encoder for transforming input features to numerical values.

    y_scaler_ : StandardScaler
        Scaler for transforming target values.

    ensemble_generator_ : EnsembleGenerator
        Fitted ensemble generator that creates multiple dataset views.

    model_ : TabICL
        The loaded TabICL model used for predictions.

    model_path_ : str
        Path to the loaded checkpoint file.

    model_config_ : dict
        Configuration dictionary from the loaded checkpoint.

    device_ : torch.device
        The device where the model is loaded and computations are performed.

    inference_config_ : InferenceConfig
        The inference configuration.

    cache_mode_ : str or None
        The resolved caching mode, set during ``fit()`` based on the ``kv_cache``
        init parameter. One of ``"kv"``, ``"repr"``, or ``None`` (no caching).

    model_kv_cache_ : OrderedDict[str, TabICLCache] or None
        Pre-computed KV caches for training data, keyed by normalization method.
        Created during ``fit()`` when ``kv_cache`` is enabled. When set,
        ``predict()`` reuses the cached key-value projections instead of
        re-processing training data, enabling faster inference on multiple test sets.
    """

    def __init__(
        self,
        n_estimators: int = 8,
        norm_methods: Optional[str | List[str]] = None,
        feat_shuffle_method: str = "latin",
        outlier_threshold: float = 4.0,
        batch_size: Optional[int] = 8,
        kv_cache: bool | str = False,
        model_path: Optional[str | Path] = None,
        allow_auto_download: bool = True,
        checkpoint_version: str = "tabicl-regressor-v2-20260212.ckpt",
        device: Optional[str | torch.device] = None,
        use_amp: bool | str = "auto",
        use_fa3: bool | str = "auto",
        offload_mode: str | bool = "auto",
        disk_offload_dir: Optional[str] = None,
        random_state: int | None = 42,
        n_jobs: Optional[int] = None,
        verbose: bool = False,
        inference_config: Optional[InferenceConfig | Dict] = None,
    ):
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.outlier_threshold = outlier_threshold
        self.batch_size = batch_size
        self.kv_cache = kv_cache
        self.model_path = model_path
        self.allow_auto_download = allow_auto_download
        self.checkpoint_version = checkpoint_version
        self.device = device
        self.use_amp = use_amp
        self.use_fa3 = use_fa3
        self.offload_mode = offload_mode
        self.disk_offload_dir = disk_offload_dir
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.inference_config = inference_config

    def _load_model(self) -> None:
        """Load a model from a given path or download it if not available.

        It uses `model_path` and `checkpoint_version` to determine the source.
         - If `model_path` is specified and exists, it's used directly.
         - If `model_path` is specified but doesn't exist (and auto-download is enabled),
           the version specified by `checkpoint_version` is downloaded to `model_path`.
         - If `model_path` is None, the version specified by `checkpoint_version` is downloaded
           from Hugging Face Hub and cached in the default Hugging Face cache directory.

        Raises
        ------
        AssertionError
            If the checkpoint doesn't contain the required 'config' or 'state_dict' keys.

        ValueError
            If a checkpoint cannot be found or downloaded based on the settings.
        """

        repo_id = "jingang/TabICL"
        filename = self.checkpoint_version

        if self.model_path is None:
            # Scenario 1: the model path is not provided, so download from HF Hub based on the checkpoint version
            try:
                model_path_ = Path(hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True))
            except LocalEntryNotFoundError:
                if self.allow_auto_download:
                    print(f"Checkpoint '{filename}' not cached.\n Downloading from Hugging Face Hub ({repo_id}).\n")
                    model_path_ = Path(hf_hub_download(repo_id=repo_id, filename=filename))
                else:
                    raise ValueError(
                        f"Checkpoint '{filename}' not cached and automatic download is disabled.\n"
                        f"Set allow_auto_download=True to download the checkpoint from Hugging Face Hub ({repo_id})."
                    )
            if model_path_:
                checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
        else:
            # Scenario 2: the model path is provided
            model_path_ = Path(self.model_path) if isinstance(self.model_path, str) else self.model_path
            if model_path_.exists():
                # Scenario 2a: the model path exists, load it directly
                checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
            else:
                # Scenario 2b: the model path does not exist, download the checkpoint version to this path
                if self.allow_auto_download:
                    print(
                        f"Checkpoint not found at '{model_path_}'.\n"
                        f"Downloading '{filename}' from Hugging Face Hub ({repo_id}) to this location.\n"
                    )
                    model_path_.parent.mkdir(parents=True, exist_ok=True)
                    cache_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=model_path_.parent)
                    Path(cache_path).rename(model_path_)
                    checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
                else:
                    raise ValueError(
                        f"Checkpoint not found at '{model_path_}' and automatic download is disabled.\n"
                        f"Either provide a valid checkpoint path, or set allow_auto_download=True to download "
                        f"'{filename}' from Hugging Face Hub ({repo_id})."
                    )

        assert "config" in checkpoint, "The checkpoint doesn't contain the model configuration."
        assert "state_dict" in checkpoint, "The checkpoint doesn't contain the model state."

        self.model_path_ = model_path_

        config = checkpoint["config"]
        self.model_ = TabICL(**config)
        self.model_config_ = config
        self.model_.load_state_dict(checkpoint["state_dict"])
        self.model_.eval() # 加载后模型设置为 eval() 模式（不计算梯度）

    def fit(self, X: np.ndarray, y: np.ndarray) -> TabICLRegressor:
        """Fit the regressor to training data.

        Prepares the model for prediction by:

        1. Scaling target values using StandardScaler
        2. Converting input features to numerical values
        3. Fitting the ensemble generator to create transformed dataset views
        4. Loading the pre-trained TabICL model
        5. Optionally pre-computing KV caches for training data to speed up inference
           (controlled by the ``kv_cache`` init parameter)

        The model itself is not trained on the data; it uses in-context learning
        at inference time. This method only prepares the data transformations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input data.

        y : array-like of shape (n_samples,)
            Training target values.

        Returns
        -------
        self : TabICLRegressor
            Fitted regressor instance.
        """

        if y is None: # 回归任务必须提供目标值 y（不像某些无监督方法）
            raise ValueError("This regressor requires y to be passed, but the target y is None.")

        # 调用 sklearn 的数据验证工具，确保 X 和 y 的样本数一致、没有全 NaN 列等
        X, y = validate_data(self, X, y, dtype=None, skip_check_array=True)

        # Ensure y is numeric
        y = np.asarray(y, dtype=np.float32) # 强制将 y 转为 float32 数组，确保后续数值计算的兼容性

        # Warn and flatten 2D column-vector y
        if y.ndim == 2 and y.shape[1] == 1: # 如果 y 是列向量 (n, 1)，展平为 (n,)。这是 sklearn 的标准兼容性处理
            from sklearn.exceptions import DataConversionWarning

            warnings.warn(
                "A column-vector y was passed when a 1d array was expected. Please change "
                "the shape of y to (n_samples, ), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )
            y = y.ravel()

        # Device setup
        self._resolve_device()

        # Inference configuration
        self.n_samples_in_ = _num_samples(X) # 记录训练样本数，后续用于 AMP/FA3 的自动策略判断
        self._build_inference_config() # 构建 InferenceConfig 对象，包含三个 transformer（COL、ROW、ICL）的推理配置

        # Load the pre-trained TabICL model
        self._load_model() # 加载预训练的 TabICL 模型
        self.model_.to(self.device_) # 将模型参数移到目标设备（GPU/CPU）

        # Scale target values
        self.y_scaler_ = StandardScaler() # StandardScaler 计算均值 μ 和标准差 σ，然后 y_scaled = (y - μ) / σ
        y_scaled = self.y_scaler_.fit_transform(y.reshape(-1, 1)).flatten()

        # Transform input features
        self.X_encoder_ = TransformToNumerical(verbose=self.verbose)
        X = self.X_encoder_.fit_transform(X) # 若数据存在非数值类型，会学习映射为数值的规则，再使用此规则执行转换，内部其实是.fit().transform()链式调用

        # Fit ensemble generator to create multiple dataset views
        # EnsembleGenerator 是 TabICL 提升预测质量的核心机制。它通过创建多个不同视角的数据视图来做集成预测
        self.ensemble_generator_ = EnsembleGenerator(
            classification=False, # 回归任务
            n_estimators=self.n_estimators, # 集成成员数量，默认8
            norm_methods=self.norm_methods or ["none", "power"], # 归一化方法
            feat_shuffle_method=self.feat_shuffle_method, # 特征排列策略，默认"latin"
            outlier_threshold=self.outlier_threshold, # 异常值阈值，默认4.0
            random_state=self.random_state, # 随机种子，默认42
        )
        self.ensemble_generator_.fit(X, y_scaled) # fit过程：为每种归一化方法拟合预处理器（记录训练数据的统计量），供后续 transform() 使用

        self.model_kv_cache_ = None
        if self.kv_cache: # kv cache 可选加速
            # 当 kv_cache=True，fit 阶段会预先计算训练数据的 Key-Value 投影并缓存，这样 predict 时就不需要重复处理训练数据，大幅加速推理
            if self.kv_cache is True or self.kv_cache == "kv":
                self.cache_mode_ = "kv"
            elif self.kv_cache == "repr":
                self.cache_mode_ = "repr"
            else:
                raise ValueError(f"Invalid kv_cache value '{self.kv_cache}'. Expected False, True, 'kv', or 'repr'.")
            self._build_kv_cache()

        return self

    def _build_kv_cache(self) -> None:
        """Pre-compute KV caches for training data across all ensemble batches."""
        
        # _build_kv_cache 将训练数据通过已拟合的 EnsembleGenerator 转换为所有集成变体视图，
        # 然后逐批送入 TabICL 模型前向传播，把每层 Transformer 对训练样本计算出的 Key-Value 投影缓存下来，
        # 后续 predict 时测试数据可复用这些缓存，避免对训练数据的重复计算，从而大幅加速推理

        # X=None is required in transform() even though it is the default value
        # because sklearn's _SetOutputMixin wraps transform() with a signature
        # that enforces X as a positional argument.
        # mode="train"：要求 EnsembleGenerator 只返回训练数据的集成视图（不包含测试数据），数据已经过对应的 PreprocessingPipeline 预处理
        train_data = self.ensemble_generator_.transform(X=None, mode="train") # 获取每种归一化方法下的训练数据视图
        # 结构：{norm_method: TabICLCache}，每种归一化方法对应一个独立的缓存，因为不同归一化方法的预处理输出不同，KV 投影也不同
        self.model_kv_cache_ = OrderedDict() # 初始化缓存字典

        # 对每种归一化方法（如 "none", "power"），分别构建其 KV 缓存
        for norm_method, (Xs, ys) in train_data.items():
            # 分批策略，避免 OOM
            batch_size = self.batch_size or Xs.shape[0] # 确定批次大小，若用户设batch_size则用它，否则把所有变体一次处理
            n_batches = int(np.ceil(Xs.shape[0] / batch_size)) # 计算需要多少个批次
            Xs_split = np.array_split(Xs, n_batches) # 沿第 0 维（变体维度）均匀切分
            ys_split = np.array_split(ys, n_batches)

            # 逐批前向传播并缓存
            caches = []
            for X_batch, y_batch in zip(Xs_split, ys_split):
                X_batch = torch.from_numpy(X_batch).float().to(self.device_)
                y_batch = torch.from_numpy(y_batch).float().to(self.device_)
                # 前向传播，存储 KV cache
                with torch.no_grad(): # 禁用梯度计算。因为是推理阶段，不需要计算梯度，节省显存和计算量
                    # 核心调用
                    self.model_.predict_stats_with_cache(
                        X_train=X_batch,
                        y_train=y_batch,
                        use_cache=False, # 不使用已有缓存（因为现在是构建缓存）
                        store_cache=True, # 将计算结果存到 self.model_._cache
                        cache_mode=self.cache_mode_, # 缓存策略 kv或repr
                        inference_config=self.inference_config_,
                    )
                caches.append(self.model_._cache)
                self.model_.clear_cache() # 清除模型内部缓存，避免下一批追加到错误的缓存上

            # Merge all batch caches into a single cache
            self.model_kv_cache_[norm_method] = TabICLCache.concat(caches) # 合并所有批次的缓存

    def _batch_forward(
        self,
        Xs: np.ndarray,
        ys: np.ndarray,
        output_type: str | list[str] = "mean",
        alphas: Optional[List[float]] = None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Process model forward passes in batches to manage memory efficiently.

        This method handles the batched inference through the TabICL model,
        dividing the ensemble members into smaller batches to avoid out-of-memory errors.

        Parameters
        ----------
        Xs : np.ndarray
            Input features of shape ``(n_datasets, n_samples, n_features)``, where
            ``n_datasets`` is the number of ensemble members.

        ys : np.ndarray
            Training labels of shape ``(n_datasets, train_size)``, where ``train_size``
            is the number of samples used for in-context learning.

        output_type : str or list of str, default="mean"
            Type of output to return (``"mean"``, ``"median"``, ``"variance"``,
            or ``"quantiles"``).

        alphas : list of float or None, default=None
            Probability levels to return if ``output_type`` includes ``"quantiles"``.

        Returns
        -------
        np.ndarray or dict[str, np.ndarray]
            Model outputs. Shape depends on ``output_type``.
        """

        batch_size = self.batch_size or Xs.shape[0]
        n_batches = np.ceil(Xs.shape[0] / batch_size)
        Xs = np.array_split(Xs, n_batches)
        ys = np.array_split(ys, n_batches)

        output_type = [output_type] if isinstance(output_type, str) else output_type
        results = {key: [] for key in output_type}

        # 这里 X_batch 包含训练+测试数据，y_batch 是训练标签。模型内部会：
        # 列级嵌入：(B, n_train+n_test, H) → (B, n_train+n_test, H+C, E)
        # 行级交互：→ (B, n_train+n_test, C*E)
        # ICL 预测：用训练部分作为上下文，预测测试部分
        for X_batch, y_batch in zip(Xs, ys):
            X_batch = torch.from_numpy(X_batch).float().to(self.device_)
            y_batch = torch.from_numpy(y_batch).float().to(self.device_)

            with torch.no_grad():
                out = self.model_.predict_stats(
                    X_batch,
                    y_batch,
                    output_type=output_type,
                    alphas=alphas,
                    inference_config=self.inference_config_,
                )
                if isinstance(out, dict):
                    for key in output_type:
                        results[key].append(out[key].float().cpu().numpy())
                else:
                    results[output_type[0]].append(out.float().cpu().numpy())

        # Concatenate batches
        for key in results:
            results[key] = np.concatenate(results[key], axis=0)

        if len(output_type) == 1:
            return results[output_type[0]]

        return results

    def _batch_forward_with_cache(
        self,
        Xs: np.ndarray,
        kv_cache: TabICLCache,
        output_type: str | list[str] = "mean",
        alphas: Optional[List[float]] = None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Process model forward passes using a pre-computed KV cache.

        The cache is sliced along the batch dimension to match each batch.

        Parameters
        ----------
        Xs : np.ndarray
            Test features of shape ``(n_datasets, test_size, n_features)``.

        kv_cache : TabICLCache
            Single KV cache for all estimators of a normalization method.

        output_type : str or list of str, default="mean"
            Type of output to return (``"mean"``, ``"median"``, ``"variance"``,
            or ``"quantiles"``).

        alphas : list of float or None, default=None
            Probability levels to return if ``output_type`` includes ``"quantiles"``.

        Returns
        -------
        np.ndarray or dict[str, np.ndarray]
            Model outputs. Shape depends on ``output_type``.
        """
        n_total = Xs.shape[0]
        batch_size = self.batch_size or n_total
        n_batches = int(np.ceil(n_total / batch_size))
        Xs_split = np.array_split(Xs, n_batches)

        output_type = [output_type] if isinstance(output_type, str) else output_type
        results = {key: [] for key in output_type}

        offset = 0
        for X_batch in Xs_split: # 分批处理避免 OOM
            bs = X_batch.shape[0]
            cache_subset = kv_cache.slice_batch(offset, offset + bs) # 从 cache 中切出当前 batch 对应的子集
            offset += bs

            X_batch = torch.from_numpy(X_batch).float().to(self.device_)
            with torch.no_grad():
                out = self.model_.predict_stats_with_cache( # 关键：predict_stats_with_cache 只需处理测试数据，训练数据的 K/V 投影从 cache 直接读取
                    X_test=X_batch,
                    output_type=output_type,
                    alphas=alphas,
                    cache=cache_subset,
                    inference_config=self.inference_config_,
                )
                if isinstance(out, dict):
                    for key in output_type:
                        results[key].append(out[key].float().cpu().numpy())
                else:
                    results[output_type[0]].append(out.float().cpu().numpy())

        # Concatenate batches
        for key in results:
            results[key] = np.concatenate(results[key], axis=0)

        if len(output_type) == 1:
            return results[output_type[0]]

        return results

    def predict(
        self, X: np.ndarray, output_type: str | list[str] = "mean", alphas: Optional[List[float]] = None
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Predict target values for test samples.

        Applies the ensemble of TabICL models to make predictions, with each ensemble
        member providing predictions that are then averaged. The method:

        1. Transforms input data using the fitted encoders
        2. Applies the ensemble generator to create multiple views
        3. Forwards each view through the model
        4. Averages predictions across ensemble members
        5. Inverse transforms predictions to original scale

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples for prediction.  Columns that are entirely NaN are
            treated as masked features and excluded from inference.  This is
            useful for computing SHAP values, where masked features are
            represented as all-NaN columns.

        output_type : str or list of str, default="mean"
            Determines the type of output to return.

            - If ``"mean"``, returns the mean over the predicted distribution.
            - If ``"median"``, returns the median over the predicted distribution.
            - If ``"quantiles"``, returns the quantiles of the predicted distribution.
              The parameter ``alphas`` determines which quantiles are returned.
            - If ``"raw_quantiles"``, returns the raw quantiles (direct outputs of TabICL).
            - If a list of str, returns multiple types of outputs as specified in the list.

        alphas : list of float or None, default=None
            The probability levels to return if ``output_type="quantiles"``.

            By default, the ``[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]``
            quantiles are returned. The predictions per quantile match
            the input order.

        Returns
        -------
        np.ndarray of shape (n_samples,) or dict[str, np.ndarray]
            An array of shape ``(n_samples,)`` if ``output_type`` is ``"mean"`` or
            ``"median"``, or an array of shape ``(n_samples, n_quantiles)`` if
            ``output_type`` is ``"quantiles"`` or ``"raw_quantiles"``.

            If ``output_type`` is a list of str, returns a dictionary with keys as
            specified in the list and values as the corresponding predictions.
        """
        check_is_fitted(self) # 检查确保已经调用过 fit()
        if isinstance(X, np.ndarray) and len(X.shape) == 1:
            # Reject 1D arrays to maintain sklearn compatibility
            raise ValueError("The provided input X is one-dimensional. Reshape your data.")

        # Check if prediction is possible
        # 预测需要训练数据作为上下文（ICL 的核心），有两种方式：kv cache和训练数据，两者都没有则无法预测（可能序列化时排除了训练数据和 cache）
        # KV Cache：fit 时预计算了训练数据的 KV 投影
        has_kv_cache = hasattr(self, "model_kv_cache_") and self.model_kv_cache_ is not None
        # 训练数据：ensemble_generator_ 中保存了原始训练数据
        has_training_data = (
            hasattr(self, "ensemble_generator_") and getattr(self.ensemble_generator_, "X_", None) is not None
        )
        if not has_kv_cache and not has_training_data:
            raise RuntimeError(
                "Cannot predict: this estimator was saved without training data and has no KV cache. "
                "Predictions require either cached KV projections or the original training data. "
                "Re-fit the estimator or load from a file saved with save_training_data=True or "
                "save_kv_cache=True."
            )

        # 线程配置（CPU 推理优化），在 CPU 推理时控制 PyTorch 的线程数，注：函数结束时会恢复原始线程数
        if self.n_jobs is not None:
            assert self.n_jobs != 0
            old_n_threads = torch.get_num_threads()
            n_logical_cores = mp.cpu_count()

            if self.n_jobs > 0: # 使用指定数量的线程（上限为逻辑核心数）
                if self.n_jobs > n_logical_cores:
                    warnings.warn(
                        f"TabICL got n_jobs={self.n_jobs} but there are only {n_logical_cores} logical cores available."
                        f" Only {n_logical_cores} threads will be used."
                    )
                n_threads = min(n_logical_cores, self.n_jobs)
            else: # 比如-1 时 = 使用全部核心
                n_threads = max(1, n_logical_cores + 1 + self.n_jobs)

            torch.set_num_threads(n_threads)

        # Preserve DataFrame structure to retain column names and types for correct feature transformation
        X = validate_data(self, X, reset=False, dtype=None, skip_check_array=True) # reset=False不重置已 fit 的属性，只验证特征数一致性

        # Detect all-NaN columns (used by SHAP's feature masking approach)
        # feature_mask 是布尔数组，True 表示该列全为 NaN（被掩码）
        if hasattr(X, "columns"):  # check for dataframe without importing pandas  有columns属性说明是DataFrame
            # isna检查每个位置是否为NaN → 布尔矩阵 (n, m)；all(axis=0)沿行方向，检查每列是否全为True → (m,)
            feature_mask = X.isna().all(axis=0).to_numpy()
        else:
            arr = np.asarray(X)
            if np.issubdtype(arr.dtype, np.number): # 是NumPy数值类型
                feature_mask = np.isnan(arr).all(axis=0)
            else: # 是NumPy对象数组（混合类型/字符串）
                # object dtype: v != v is True only for NaN in IEEE 754, safe for strings too
                feature_mask = np.array([all(v != v for v in arr[:, i]) for i in range(arr.shape[1])])

        if feature_mask is not None and not np.any(feature_mask):
            feature_mask = None

        # Fill masked columns so that transformers don't choke on NaN
        if feature_mask is not None: # 将全 NaN 列填充为 0，防止后续 sklearn transformer 报错
            if hasattr(X, "columns"):  # Proxy way to check whether X is a dataframe
                X.iloc[:, feature_mask] = 0.0
            else:
                X[:, feature_mask] = 0.0

        X = self.X_encoder_.transform(X) # 使用 fit 时拟合的 TransformToNumerical 对测试数据做相同变换

        # 输出类型标准化。统一转为列表，方便后续处理多种输出类型
        output_type = [output_type] if isinstance(output_type, str) else list(output_type)

        # 集成推理（核心分支）

        # Skip KV cache when features are masked
        # 这一行重复前面的，其实多余，第一次用于检查能否预测，第二次(即这一行)用于决定是否使用 cache
        has_kv_cache = hasattr(self, "model_kv_cache_") and self.model_kv_cache_ is not None
        use_cache = has_kv_cache and feature_mask is None

        if use_cache:
            # X (n_test, n_features)
            # ↓ ensemble_generator_.transform(mode="test") 为每种 norm_method 生成特征排列视图
            # OrderedDict {
            #   "none":  (Xs_test: [n_estimators, n_test, n_features],),
            #   "power": (Xs_test: [n_estimators, n_test, n_features],),
            # }
            # ↓ 对每种 norm_method: 取出对应的 kv_cache（fit 时预计算的训练数据 KV 投影）
            # _batch_forward_with_cache(Xs_test, kv_cache) → 只前向传播测试数据，复用训练数据的 cache

            # Cache exists: forward only test data and use the pre-computed cache for training data
            test_data = self.ensemble_generator_.transform(X, mode="test") # 只处理测试数据，生成每种 norm_method 的特征排列视图
            results = {key: [] for key in output_type}
            for norm_method, (Xs_test,) in test_data.items():
                kv_cache = self.model_kv_cache_[norm_method] # 取出预计算缓存
                batch_out = self._batch_forward_with_cache(Xs_test, kv_cache, output_type=output_type, alphas=alphas) # 只前向测试数据
                if isinstance(batch_out, dict):
                    for key in output_type:
                        results[key].append(batch_out[key])
                else:
                    results[output_type[0]].append(batch_out)
        else:
            # No cache or masked features: forward both training and test data
            data = self.ensemble_generator_.transform(X, mode="both", feature_mask=feature_mask) # mode="both" 返回训练 + 测试数据拼接后的视图
            results = {key: [] for key in output_type}
            for Xs, ys in data.values():
                batch_out = self._batch_forward(Xs, ys, output_type=output_type, alphas=alphas)
                if isinstance(batch_out, dict):
                    for key in output_type:
                        results[key].append(batch_out[key])
                else:
                    results[output_type[0]].append(batch_out)

        # 反标准化与集成平均
        # Concatenate across ensemble members and apply inverse transform
        final_results = {}
        for key in output_type:
            arr = np.concatenate(results[key], axis=0) # 拼接所有 norm_method 的预测结果
            n_estimators = arr.shape[0]
            n_samples = arr.shape[1]

            if arr.ndim == 2:
                # mean, variance, or median: (n_estimators, n_samples)
                arr = self.y_scaler_.inverse_transform(arr.reshape(-1, 1)).reshape(n_estimators, n_samples)
                # 将标准化空间的预测还原到原始尺度：y = y_scaled × σ + μ，其中 σ 和 μ 是 fit 时 StandardScaler 计算的标准差和均值
                final_results[key] = np.mean(arr, axis=0) # 所有集成成员取平均 → 最终预测
            else:
                # quantiles: (n_estimators, n_samples, n_quantiles)
                n_quantiles = arr.shape[2]
                arr = self.y_scaler_.inverse_transform(arr.reshape(-1, 1)).reshape(n_estimators, n_samples, n_quantiles)
                final_results[key] = np.mean(arr, axis=0)

        # 恢复 PyTorch 原始线程数
        if self.n_jobs is not None:
            torch.set_num_threads(old_n_threads)

        # 单一输出类型 → 直接返回数组；多种输出类型 → 返回字典
        if len(output_type) == 1:
            return final_results[output_type[0]]

        return final_results

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
