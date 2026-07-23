from __future__ import annotations
from typing import Optional, List, Union, Literal

import torch
from torch import nn, Tensor

from .embedding import ColEmbedding
from .interaction import RowInteraction
from .learning import ICLearning
from .action_encoder import ActionEncoder
from .quantile_dist import QuantileToDistribution
from .kv_cache import TabICLCache
from .inference_config import InferenceConfig


class TabICL(nn.Module):
    """A Tabular In-Context Learning Foundation Model.

    TabICL is a transformer-based architecture for in-context learning on tabular data to make
    predictions without fine-tuning. It processes tabular data through three sequential stages:

    1. Column-wise embedding creates distribution-aware embeddings
    2. Row-wise interaction captures interactions between features within each row
    3. Dataset-wise in-context learning to learn patterns from labeled examples and make predictions

    This class is the underlying raw PyTorch module for TabICL. It is not
    intended to be used directly. Instead, use the classes from the top-level
    `tabicl` package such as :class:`tabicl.TabICLClassifier` or
    :class:`tabicl.TabICLRegressor` that wrap this class to include the
    necessary preprocessing of input features and postprocessing of
    predictions.

    Parameters
    ----------
    max_classes : int, default=10
        Determines the task type and output behavior:
        - If max_classes=0: The model performs regression using quantile prediction.
        - If max_classes>0: The model performs classification. This value specifies
          the number of classes the model supports natively. If the number of classes
          in the dataset exceeds this value, mixed-radix ensembling is used during
          column-wise embedding and hierarchical classification is used during in-context learning.

    num_quantiles : int, default=999
        Number of quantiles to predict for regression tasks. Only used when max_classes=0.
        The model directly predicts these quantile values.

    embed_dim : int, default=128
        Model dimension used in the column / row embedding transformers. For the in-context
        learning transformer, the dimension is this value multiplied by the number of CLS tokens.

    col_num_blocks : int, default=3
        Number of induced self-attention blocks in the column embedding transformer.

    col_nhead : int, default=8
        Number of attention heads in the column embedding transformer.

    col_num_inds : int, default=128
        Number of inducing points in the column embedding transformer.

    col_affine : bool, default=False
        If True, computes embeddings as: :math:`\\text{features} \\times W + b`.
        If False, directly uses the set transformer output as embeddings.

    col_feature_group : bool or Literal["same", "valid"], default="same"
        Feature grouping mode:
        - False: No grouping
        - True or "same": Group through circular permutation (output has same number of groups as features)
        - "valid": Group through padding and reshaping (output may have fewer groups)

    col_feature_group_size : int, default=3
        Number of features per group when feature grouping is enabled.

    col_target_aware : bool, default=True
        If True, incorporates target information into column-wise embeddings.

    col_ssmax : bool or str, default="qassmax-mlp-elementwise"
        Type of scalable softmax to use in the column embedding transformer. Note that only the first
        attention layer of the induced self-attention blocks uses SSMax.
        If True, equivalent to "qassmax-mlp-elementwise".
        If False, equivalent to "none".
        If a string, uses the specified scalable softmax type.
        Options include:
            - "none": No scaling applied
            - "ssmax": :math:`q_{\\text{scaled}} = q \\cdot (s \\cdot \\log n)` where s is learnable per-head parameter
            - "ssmax-mlp": Uses MLP to compute scaling factors based on sequence length
            - "ssmax-mlp-elementwise": Elementwise scaling per head dimension using MLP
            - "qassmax-mlp": Query-aware scaling: :math:`\\text{scale} = \\text{base\\_mlp}(\\log n) \\cdot (1 + \\tanh(\\text{query\\_mlp}(q)))`
            - "qassmax-mlp-elementwise": Elementwise query-aware scaling

    row_num_blocks : int, default=3
        Number of attention blocks in the row interaction transformer.

    row_nhead : int, default=8
        Number of attention heads in the row interaction transformer.

    row_num_cls : int, default=4
        Number of learnable CLS tokens used to aggregate feature information per row.

    row_rope_base : float, default=100000
        Base scaling factor for rotary position encoding in the row interaction transformer.

    row_rope_interleaved : bool, default=False
        If True, uses interleaved rotation where dimension pairs are (0,1), (2,3), etc.
        If False, uses non-interleaved rotation where the embedding is split into
        first half [0:d//2] and second half [d//2:d].

    icl_num_blocks : int, default=12
        Number of transformer blocks in the in-context learning transformer.

    icl_nhead : int, default=8
        Number of attention heads in the in-context learning transformer.

    icl_ssmax : bool or str, default="qassmax-mlp-elementwise"
        Type of scalable softmax to use in the in-context learning transformer.
        If True, equivalent to "qassmax-mlp-elementwise".
        If False, equivalent to "none".
        If a string, uses the specified scalable softmax type.
        Options include:
            - "none": No scaling applied
            - "ssmax": :math:`q_{\\text{scaled}} = q \\cdot (s \\cdot \\log n)` where s is learnable per-head parameter
            - "ssmax-mlp": Uses MLP to compute scaling factors based on sequence length
            - "ssmax-mlp-elementwise": Elementwise scaling per head dimension using MLP
            - "qassmax-mlp": Query-aware scaling: :math:`\\text{scale} = \\text{base\\_mlp}(\\log n) \\cdot (1 + \\tanh(\\text{query\\_mlp}(q)))`
            - "qassmax-mlp-elementwise": Elementwise query-aware scaling

    ff_factor : int, default=2
        Expansion factor for feedforward networks across all components.

    dropout : float, default=0.0
        Dropout probability across all components.

    activation : str or unary callable, default="gelu"
        Activation function used throughout the model.

    norm_first : bool, default=True
        If True, uses pre-norm architecture across all components.

    bias_free_ln : bool, default=False
        If True, removes bias from all LayerNorm layers (sets bias=False in nn.LayerNorm).

    recompute : bool, default=False
        If True, uses gradient checkpointing to save memory at the cost of additional computation.
    """

    def __init__(
        self,
        max_classes: int = 10,
        num_quantiles: int = 999,
        embed_dim: int = 128,
        col_num_blocks: int = 3,
        col_nhead: int = 8,
        col_num_inds: int = 128,
        col_affine: bool = False,
        col_feature_group: Union[bool, Literal["same", "valid"]] = "same",
        col_feature_group_size: int = 3,
        col_target_aware: bool = True,
        col_ssmax: Union[ # 参数名 + 类型声明开始
            bool, # 第一种允许的类型
            Literal[ # 第二种类型：精确字面值列表。允许的字符串值（6选1）
                "none",
                "ssmax",
                "ssmax-mlp",
                "ssmax-mlp-elementwise",
                "qassmax-mlp",
                "qassmax-mlp-elementwise",
            ],
        ] = "qassmax-mlp-elementwise", # 默认值
        row_num_blocks: int = 3,
        row_nhead: int = 8,
        row_num_cls: int = 4,
        row_rope_base: float = 100000,
        row_rope_interleaved: bool = False,
        icl_num_blocks: int = 12,
        icl_nhead: int = 8,
        icl_ssmax: Union[
            bool,
            Literal[
                "none",
                "ssmax",
                "ssmax-mlp",
                "ssmax-mlp-elementwise",
                "qassmax-mlp",
                "qassmax-mlp-elementwise",
            ],
        ] = "qassmax-mlp-elementwise",
        ff_factor: int = 2,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        recompute: bool = False,
        num_outputs: int = 1, # 输出维度数，>1 时共享 backbone，每维度独立 y_encoder + decoder head
        # 双流模式（实验 1/2）：分离 state 和 action 处理
        use_action_encoder: bool = False, # 启用独立的 ActionEncoder
        state_dim: int = 11,              # 状态维度（Hopper-v5: 11）
        action_dim: int = 3,              # 动作维度（Hopper-v5: 3）
        action_encoder_mode: Literal["mlp", "transformer"] = "mlp", # ActionEncoder 类型
    ):
        super().__init__()
        icl_dim = embed_dim * row_num_cls  # CLS tokens are concatenated for ICL

        # Determine task type
        if max_classes == 0:  # Regression
            if num_quantiles <= 0:
                raise ValueError("For regression (max_classes=0), num_quantiles must be greater than 0.")
            out_dim = num_quantiles
            self.quantile_dist = QuantileToDistribution(num_quantiles=num_quantiles)
        else:  # Classification
            out_dim = max_classes

        self.max_classes = max_classes
        self.num_quantiles = num_quantiles
        self.num_outputs = num_outputs
        self.use_action_encoder = use_action_encoder
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.col_num_blocks = col_num_blocks
        self.col_nhead = col_nhead
        self.col_num_inds = col_num_inds
        self.col_affine = col_affine
        self.col_feature_group = col_feature_group
        self.col_feature_group_size = col_feature_group_size
        self.col_target_aware = col_target_aware
        self.col_ssmax = col_ssmax
        self.row_num_blocks = row_num_blocks
        self.row_nhead = row_nhead
        self.row_num_cls = row_num_cls
        self.row_rope_base = row_rope_base
        self.row_rope_interleaved = row_rope_interleaved
        self.icl_num_blocks = icl_num_blocks
        self.icl_nhead = icl_nhead
        self.icl_ssmax = icl_ssmax
        self.ff_factor = ff_factor
        self.dropout = dropout
        self.activation = activation
        self.norm_first = norm_first
        self.bias_free_ln = bias_free_ln

        self.col_embedder = ColEmbedding(
            embed_dim=embed_dim,
            num_blocks=col_num_blocks,
            nhead=col_nhead,
            num_inds=col_num_inds,
            dim_feedforward=embed_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            affine=col_affine,
            feature_group=col_feature_group,
            feature_group_size=col_feature_group_size,
            target_aware=col_target_aware,
            max_classes=max_classes,
            reserve_cls_tokens=row_num_cls,
            ssmax=col_ssmax,
            recompute=recompute,
        )

        self.row_interactor = RowInteraction(
            embed_dim=embed_dim,
            num_blocks=row_num_blocks,
            nhead=row_nhead,
            dim_feedforward=embed_dim * ff_factor,
            num_cls=row_num_cls,
            rope_base=row_rope_base,
            rope_interleaved=row_rope_interleaved,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            recompute=recompute,
        )

        self.icl_predictor = ICLearning(
            out_dim=out_dim,
            max_classes=max_classes,
            d_model=icl_dim,
            num_blocks=icl_num_blocks,
            nhead=icl_nhead,
            dim_feedforward=icl_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            ssmax=icl_ssmax,
            recompute=recompute,
            num_outputs=num_outputs, # 多输出回归任务：共享 ICL backbone，每维度独立 head
        )

        # 双流模式：独立的 ActionEncoder + 融合层
        if use_action_encoder:
            self.action_encoder = ActionEncoder(
                action_dim=action_dim,
                d_model=embed_dim,
                hidden_dim=embed_dim * 2, # 256
                mode=action_encoder_mode,
            )
            # 融合投影：将 [state_repr(C*E) | action_repr(E)] → ICL 期望的 d_model(C*E)
            self.fusion_proj = nn.Linear(icl_dim + embed_dim, icl_dim)
        else:
            self.action_encoder = None
            self.fusion_proj = None

        # KV cache for efficient inference
        self._cache: Optional[TabICLCache] = None

    @property
    def has_cache(self) -> bool:
        """Check if a valid cache is stored."""
        return self._cache is not None and not self._cache.is_empty()

    def clear_cache(self) -> None:
        """Clear the stored cache."""
        self._cache = None

    def _train_forward(
        self, X: Tensor, y_train: Tensor, d: Optional[Tensor] = None, embed_with_test: bool = False
    ) -> Tensor:
        """Column-wise embedding -> row-wise interaction -> dataset-wise in-context learning for training.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)
            The first train_size positions contain training samples, and the remaining positions contain test samples.

        y_train : Tensor
            Training labels.
            - Single-output: shape (B, train_size)
            - Multi-output: shape (B, train_size, num_outputs)

        d : Optional[Tensor], default=None
            The number of features per dataset.

        embed_with_test : bool, default=False
            If True, allow training samples to attend to test samples during embedding.

        Returns
        -------
        Tensor
            - Single-output: shape (B, test_size, out_dim)
            - Multi-output: shape (B, test_size, num_outputs, out_dim)
        """

        B, T, H = X.shape
        # y_train.shape[1] 在单输出 (B, train_size) 和多输出 (B, train_size, num_outputs) 下都是 train_size
        train_size = y_train.shape[1]
        assert train_size <= T, "Number of training samples exceeds total samples"

        # Check if d is provided and has the same length as the number of features
        if d is not None and len(d.unique()) == 1 and d[0] == H:
            d = None

        # 多输出时：ColEmbedding 仅使用第一个输出维度做 target-aware embedding
        y_col = y_train[:, :, 0] if y_train.dim() == 3 else y_train

        if self.use_action_encoder:
            # 双流模式：state 和 action 分离处理
            X_state = X[:, :, :self.state_dim]   # (B, T, state_dim)
            X_action = X[:, :, self.state_dim:]  # (B, T, action_dim)

            # 流 1: state → ColEmbedding → RowInteraction
            state_repr = self.row_interactor(
                self.col_embedder(X_state, y_train=y_col, d=d, embed_with_test=embed_with_test),
                d=d,
            )  # (B, T, C*E)

            # 流 2: action → ActionEncoder
            action_repr = self.action_encoder(X_action)  # (B, T, E)

            # 融合: concat + Linear projection
            combined = torch.cat([state_repr, action_repr], dim=-1)  # (B, T, C*E + E)
            combined = self.fusion_proj(combined)  # (B, T, C*E)

            return self.icl_predictor(combined, y_train=y_train)
        else:
            # 原始模式: state+action 作为统一特征输入
            representations = self.row_interactor(
                self.col_embedder(X, y_train=y_col, d=d, embed_with_test=embed_with_test),
                d=d,
            )
            return self.icl_predictor(representations, y_train=y_train)

    def _inference_forward(
        self,
        X: Tensor,
        y_train: Tensor,
        feature_shuffles: Optional[List[List[int]]] = None,
        embed_with_test: bool = False,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        inference_config: Optional[InferenceConfig] = None,
    ) -> Tensor:
        """Column-wise embedding -> row-wise interaction -> dataset-wise in-context learning.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)
            The first train_size positions contain training samples, and the remaining positions contain test samples.

        y_train : Tensor
            Training labels of shape (B, train_size) where:
             - B is the number of tables
             - train_size is the number of training samples provided for in-context learning

        feature_shuffles : Optional[List[List[int]]], default=None
            A list of feature shuffle patterns for each table in the batch.
            When provided, indicates that X contains the same table with different feature orders.
            In this case, column-wise embeddings are computed once and then shuffled accordingly.

        embed_with_test : bool, default=False
            If True, allow training samples to attend to test samples during embedding.

        return_logits : bool, default=True
            If True, return raw logits instead of probabilities.

        softmax_temperature : float, default=0.9
            Temperature for the softmax function.

        inference_config : Optional[InferenceConfig], default=None
            Inference configuration.

        Returns
        -------
        Tensor
            For regression (max_classes=0):
                Predictions of shape (B, test_size, num_quantiles), where test_size = T - train_size

            For classification (max_classes>0):
                If return_logits=True: Logits of shape (B, test_size, num_classes)
                If return_logits=False: Probabilities of shape (B, test_size, num_classes)
        """

        # y_train.shape[1] 在单输出 (B, train_size) 和多输出 (B, train_size, num_outputs) 下都是 train_size
        train_size = y_train.shape[1]
        assert train_size <= X.shape[1], "Number of training samples exceeds total samples"

        if inference_config is None:
            inference_config = InferenceConfig()

        # 多输出时：ColEmbedding 仅使用第一个输出维度做 target-aware embedding
        y_col = y_train[:, :, 0] if y_train.dim() == 3 else y_train

        if self.use_action_encoder:
            # 双流模式
            X_state = X[:, :, :self.state_dim]
            X_action = X[:, :, self.state_dim:]

            state_repr = self.row_interactor(
                self.col_embedder(
                    X_state, y_train=y_col,
                    embed_with_test=embed_with_test,
                    feature_shuffles=feature_shuffles,
                    mgr_config=inference_config.COL_CONFIG,
                ),
                mgr_config=inference_config.ROW_CONFIG,
            )

            action_repr = self.action_encoder(X_action)
            combined = torch.cat([state_repr, action_repr], dim=-1)
            combined = self.fusion_proj(combined)
        else:
            # 原始模式
            combined = self.row_interactor(
                self.col_embedder(
                    X, y_train=y_col,
                    embed_with_test=embed_with_test,
                    feature_shuffles=feature_shuffles,
                    mgr_config=inference_config.COL_CONFIG,
                ),
                mgr_config=inference_config.ROW_CONFIG,
            )

        # Dataset-wise in-context learning（多输出时传入完整 y_train）
        out = self.icl_predictor(
            combined,
            y_train=y_train,
            return_logits=return_logits,
            softmax_temperature=softmax_temperature,
            mgr_config=inference_config.ICL_CONFIG,
        )

        return out

    def forward(
        self,
        X: Tensor,
        y_train: Tensor,
        d: Optional[Tensor] = None,
        embed_with_test: bool = False,
        feature_shuffles: Optional[List[List[int]]] = None,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        inference_config: Optional[InferenceConfig] = None,
    ) -> Tensor:
        """Column-wise embedding -> row-wise interaction -> dataset-wise in-context learning.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)
            The first train_size positions contain training samples, and the remaining positions contain test samples.

        y_train : Tensor
            Training labels of shape (B, train_size) where:
             - B is the number of tables
             - train_size is the number of training samples provided for in-context learning

        d : Optional[Tensor], default=None
            The number of features per dataset. Used only in training mode.

        embed_with_test : bool, default=False
            If True, allow training samples to attend to test samples during embedding.

        feature_shuffles : Optional[List[List[int]]], default=None
            A list of feature shuffle patterns for each table in the batch. Used only in inference mode.
            When provided, indicates that X contains the same table with different feature orders.
            In this case, column-wise embeddings are computed once and then shuffled accordingly.

        return_logits : bool, default=True
            If True, return raw logits instead of probabilities. Used only in inference mode.

        softmax_temperature : float, default=0.9
            Temperature for the softmax function. Used only in inference mode.

        inference_config : Optional[InferenceConfig], default=None
            Inference configuration. Used only in inference mode.

        Returns
        -------
        Tensor
            For training mode:
                Predictions of shape (B, test_size, out_dim):

                - For regression (max_classes=0): out_dim = num_quantiles
                - For classification (max_classes>0): out_dim = max_classes

            For inference mode:
                For regression (max_classes=0):
                    Predictions of shape (B, test_size, num_quantiles)

                For classification (max_classes>0):
                    If return_logits=True: Logits of shape (B, test_size, num_classes)
                    If return_logits=False: Probabilities of shape (B, test_size, num_classes)
        """

        if self.training:
            out = self._train_forward(X, y_train, d=d, embed_with_test=embed_with_test)
        else:
            out = self._inference_forward(
                X,
                y_train,
                feature_shuffles=feature_shuffles,
                embed_with_test=embed_with_test,
                return_logits=return_logits,
                softmax_temperature=softmax_temperature,
                inference_config=inference_config,
            )

        return out

    def predict_stats(
        self,
        X: Tensor,
        y_train: Tensor,
        output_type: str = "mean",
        alphas: Optional[List[float]] = None,
        embed_with_test: bool = False,
        inference_config: InferenceConfig = None,
    ) -> Tensor:
        """Compute summary statistics from predicted quantiles.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)
            The first train_size positions contain training samples, and the remaining
            positions contain test samples.

        y_train : Tensor
            Training labels of shape (B, train_size) where:
             - B is the number of tables
             - train_size is the number of training samples provided for in-context learning

        output_type : str or list of str, default="mean"
            Determines the type of output to return. Supported values:
            - "mean": Mean of the predicted quantiles (fast, no tail modeling).
            - "variance": Variance of the predicted quantiles (fast, no tail modeling).
            - "median": Median via inverse CDF interpolation.
            - "quantiles": Specific quantiles via inverse CDF. Use `alphas` to specify levels.
            If a list, returns a dict with the requested statistics.

        alphas : Optional[List[float]], default=None
            Probability levels for quantile output. Only used when "quantiles" is in `output_type`.
            Default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].

        embed_with_test : bool, default=False
            If True, allow training samples to attend to test samples during embedding.

        inference_config : InferenceConfig
            Inference configuration.

        Returns
        -------
        Tensor or dict of Tensors
            - If `output_type` is a single string: returns the corresponding tensor.
            - If `output_type` is a list: returns a dict mapping names to tensors.

            Output shapes:

            - "mean", "variance", "median": (B, test_size)
            - "quantiles": (B, test_size, len(alphas))
            - "raw_quantiles": (B, test_size, num_quantiles), where `num_quantiles` denotes 
                the number of quantile levels configured in the model architecture.
        """
        assert self.max_classes == 0, "predict_stats is only applicable for regression tasks"

        raw_quantiles = self._inference_forward(
            X, y_train, embed_with_test=embed_with_test, inference_config=inference_config
        )  # (B, test_size, num_quantiles)

        dist = self.quantile_dist(raw_quantiles)
        raw_quantiles = dist.quantiles  # dist ensures that quantiles are monotonic

        output_type = [output_type] if isinstance(output_type, str) else output_type
        results = {}

        if "mean" in output_type:
            results["mean"] = raw_quantiles.mean(dim=-1)
        if "variance" in output_type:
            results["variance"] = raw_quantiles.var(dim=-1)
        if "median" in output_type:
            results["median"] = dist.icdf(
                alpha=torch.tensor(0.5, device=raw_quantiles.device, dtype=raw_quantiles.dtype)
            )
        if "quantiles" in output_type:
            if alphas is None:
                alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            results["quantiles"] = dist.icdf(
                alpha=torch.tensor(alphas, device=raw_quantiles.device, dtype=raw_quantiles.dtype)
            )
        if "raw_quantiles" in output_type:
            results["raw_quantiles"] = raw_quantiles

        if len(output_type) == 1:
            return results[output_type[0]]

        return results

    def forward_with_cache(
        self,
        X_train: Optional[Tensor] = None,
        y_train: Optional[Tensor] = None,
        X_test: Optional[Tensor] = None,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        use_cache: bool = False,
        store_cache: bool = True,
        cache: Optional[TabICLCache] = None,
        cache_mode: str = "kv",
        inference_config: Optional[InferenceConfig] = None,
    ) -> Optional[Tensor]:
        """Forward pass with caching support for efficient inference.

        This method enables caching of training data computations to speed up
        repeated inference on the same training context. Two caching modes are
        supported:

        - ``"kv"``: Cache KV projections from both column embedding and ICL
          transformer layers. Fastest inference but uses more memory.
        - ``"repr"``: Cache column embedding KV projections and row interaction
          outputs (representations with y_train baked in). Uses ~24x less memory
          for the ICL part, at the cost of re-running the ICL transformer.

        Exactly one of `use_cache` or `store_cache` must be True.

        When ``store_cache=True``:
        - Requires X_train and y_train to be provided
        - Processes training data and stores cached values in self._cache
        - If X_test is also provided, returns predictions for test samples
        - If X_test is None, returns None (cache-only mode)

        When ``use_cache=True``:
        - Requires X_test and a populated self._cache
        - Uses cached values for training data

        Parameters
        ----------
        X_train : Optional[Tensor], default=None
            Training input of shape (B, train_size, H). Required when store_cache=True.

        y_train : Optional[Tensor], default=None
            Training target of shape (B, train_size). Required when store_cache=True.

        X_test : Optional[Tensor], default=None
            Test input of shape (B, test_size, H). Required when use_cache=True and optional
            when store_cache=True.

        return_logits : bool, default=True
            If True, return raw logits instead of probabilities.

        softmax_temperature : float, default=0.9
            Temperature for the softmax function.

        use_cache : bool, default=False
            Whether to use cached values to avoid redundant computation.

        store_cache : bool, default=True
            Whether to store computed values in cache.

        cache : Optional[TabICLCache], default=None
            External cache to use for inference. If provided, equivalent to
            setting use_cache=True and store_cache=False, but uses the provided
            cache instead of the model's internal self._cache.

        cache_mode : str, default="kv"
            Caching strategy: ``"kv"`` for KV projection caching, ``"repr"`` for
            representation caching. Ignored when ``use_cache=True`` (auto-detected
            from cache contents).

        inference_config : Optional[InferenceConfig], default=None
            Inference configuration.

        Returns
        -------
        Optional[Tensor]
            Predictions of shape (B, test_size, out_dim), or None if store_cache=True
            and X_test is not provided.

        Raises
        ------
        ValueError
            If use_cache == store_cache (exactly one must be True),
            if store_cache=True but X_train or y_train is None, or
            if use_cache=True but X_test is None or no cache exists.
        """

        # 如果外部传入了 cache，自动切换到使用缓存模式
        if cache is not None:
            use_cache = True
            store_cache = False
            self._cache = cache

        # 互斥约束：不能同时为 True 或同时为 False
        if use_cache == store_cache:
            raise ValueError("Exactly one of use_cache or store_cache must be True")

        if cache_mode not in ("kv", "repr"):
            raise ValueError(f"cache_mode must be 'kv' or 'repr', got '{cache_mode}'")

        if inference_config is None:
            inference_config = InferenceConfig()

        # Auto-detect cache mode from cache contents
        # 使用缓存时自动检测缓存类型
        if use_cache and self._cache is not None and self._cache.cache_type == "repr":
            cache_mode = "repr"

        if store_cache:
            if X_train is None or y_train is None:
                raise ValueError("X_train and y_train are required when store_cache=True")

            # Initialize cache based on training data
            num_classes = len(torch.unique(y_train[0])) if self.max_classes > 0 else 0 # 由max_classes可决定是分类/回归任务
            self._cache = TabICLCache(train_shape=X_train.shape, num_classes=num_classes)

            if X_test is None:
                X = X_train # 只编码训练数据（纯缓存模式）
            else:
                X = torch.cat([X_train, X_test], dim=1) # 拼接训练+测试

        if use_cache:
            if X_test is None:
                raise ValueError("X_test is required when use_cache=True")

            if self._cache is None or self._cache.is_empty():
                raise ValueError("No cache available. Call with store_cache=True first.")

            X = X_test # 只需要测试数据
            y_train = None # 标签从缓存中隐式获取

        # 列嵌入 + 行交互
        # Column-wise embedding with cache support -> Row-wise interaction
        representations = self.row_interactor(
            self.col_embedder.forward_with_cache(
                X,
                col_cache=self._cache.col_cache,
                y_train=y_train,
                use_cache=use_cache,
                store_cache=store_cache,
                mgr_config=inference_config.COL_CONFIG,
            ),
            mgr_config=inference_config.ROW_CONFIG,
        )

        # ICL 学习，根据 cache_mode 分两条路径
        # kv——缓存内容：ColEmbedding 和 ICL 模块中每层 Transformer 的 Key/Value 投影；特点：灵活，测试数据可以任意长度
        # repr——缓存内容：训练样本的最终行表示（row representation）；特点：更轻量，但灵活性略差
        # Dataset-wise in-context learning
        if cache_mode == "repr":
            # 存缓存时：将 y_train 编码进训练样本的行表示中，存到 _cache.row_repr，然后返回 None
            if store_cache:
                train_size = y_train.shape[1]
                # Bake y_train into train portion of representations
                representations = self.icl_predictor.prepare_repr_cache(representations, y_train)
                self._cache.row_repr = representations[:, :train_size] # 只存训练部分

                if X_test is None:
                    return None
            # 用缓存时：从缓存取训练表示 + 拼接新计算的测试表示，传给 forward_with_repr_cache（只对 ICL 部分做前向传播）
            else:
                # Concatenate cached train representations with test representations
                train_repr = self._cache.row_repr
                train_size = train_repr.shape[1]
                representations = torch.cat([train_repr.to(representations.device), representations], dim=1)

            out = self.icl_predictor.forward_with_repr_cache(
                representations,
                train_size=train_size,
                num_classes=self._cache.num_classes,
                return_logits=return_logits,
                softmax_temperature=softmax_temperature,
                mgr_config=inference_config.ICL_CONFIG,
            )
        else:
            out = self.icl_predictor.forward_with_cache(
                representations,
                icl_cache=self._cache.icl_cache,
                y_train=y_train,
                num_classes=self._cache.num_classes,
                return_logits=return_logits,
                softmax_temperature=softmax_temperature,
                use_cache=use_cache,
                store_cache=store_cache,
                mgr_config=inference_config.ICL_CONFIG,
            )

            if X_test is None:
                return None

        return out

    def predict_stats_with_cache(
        self,
        X_train: Optional[Tensor] = None,
        y_train: Optional[Tensor] = None,
        X_test: Optional[Tensor] = None,
        output_type: str = "mean",
        alphas: Optional[List[float]] = None,
        use_cache: bool = False,
        store_cache: bool = True,
        cache: Optional[TabICLCache] = None,
        cache_mode: str = "kv",
        inference_config: Optional[InferenceConfig] = None,
    ) -> Optional[Tensor]:
        """Compute summary statistics from predicted quantiles with KV caching.

        Parameters
        ----------
        X_train : Optional[Tensor], default=None
            Training input of shape (B, train_size, H). Required when store_cache=True.

        y_train : Optional[Tensor], default=None
            Training target of shape (B, train_size). Required when store_cache=True.

        X_test : Optional[Tensor], default=None
            Test input of shape (B, test_size, H). Required when use_cache=True and
            optional when store_cache=True.

        output_type : str or list of str, default="mean"
            Determines the type of output to return. Supported values:
            - "mean": Mean of the predicted quantiles (fast, no tail modeling).
            - "variance": Variance of the predicted quantiles (fast, no tail modeling).
            - "median": Median via inverse CDF interpolation.
            - "quantiles": Specific quantiles via inverse CDF. Use `alphas` to specify levels.
            If a list, returns a dict with the requested statistics.

        alphas : Optional[List[float]], default=None
            Probability levels for quantile output. Only used when "quantiles" is in
            `output_type`. Default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].

        use_cache : bool, default=False
            Whether to use cached values to avoid redundant computation.

        store_cache : bool, default=True
            Whether to store computed values in cache.

        cache : Optional[TabICLCache], default=None
            External cache to use for inference. If provided, equivalent to
            setting use_cache=True and store_cache=False.

        cache_mode : str, default="kv"
            Caching strategy: ``"kv"`` for KV projection caching, ``"repr"`` for
            representation caching. Ignored when ``use_cache=True`` (auto-detected
            from cache contents).

        inference_config : Optional[InferenceConfig], default=None
            Inference configuration.

        Returns
        -------
        Tensor or dict of Tensors or None
            None if store_cache=True and X_test is not provided. Otherwise:

            - If `output_type` is a single string: returns the corresponding tensor.
            - If `output_type` is a list: returns a dict mapping names to tensors.

            Output shapes:

            - "mean", "variance", "median": (B, test_size)
            - "quantiles": (B, test_size, len(alphas))
            - "raw_quantiles": (B, test_size, num_quantiles), where `num_quantiles` denotes 
                the number of quantile levels configured in the model architecture.
        """
        
        # 此函数是回归任务专用的推理入口，
        # 底层委托 forward_with_cache 完成带 KV 缓存的三阶段前向传播（列嵌入 → 行交互 → ICL 学习）得到原始分位数预测，
        # 然后将分位数通过 QuantileToDistribution 转换为单调的概率分布，最后从中提取用户要求的统计量（均值/方差/中位数/指定分位数）
        
        # 此函数只用于回归
        assert self.max_classes == 0, "predict_stats_with_cache is only applicable for regression tasks"

        # 核心调用
        raw_quantiles = self.forward_with_cache(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            use_cache=use_cache,
            store_cache=store_cache,
            cache=cache,
            cache_mode=cache_mode,
            inference_config=inference_config,
        )

        # 纯缓存模式（store_cache=True, X_test=None）→ 返回值raw_quantiles=None，当前函数也返回None, 用于 _build_kv_cache
        if raw_quantiles is None:
            return None

        # 确保分位数单调性
        dist = self.quantile_dist(raw_quantiles)
        raw_quantiles = dist.quantiles

        # 统一转为列表，方便统一处理。支持多个统计量同时请求，如 ["mean", "variance", "quantiles"]
        output_type = [output_type] if isinstance(output_type, str) else output_type
        results = {}

        if "mean" in output_type:
            # 均值：对 999 个分位数取算术平均 → 期望值。最快，无尾部分布建模
            results["mean"] = raw_quantiles.mean(dim=-1) # raw_quantiles: (B, test_size, 999) → .mean(dim=-1) → (B, test_size)
        if "variance" in output_type:
            # 方差：分位数分布的方差 → 预测不确定性的度量
            results["variance"] = raw_quantiles.var(dim=-1) # 形状变化同上
        if "median" in output_type:
            # 中位数：dist.icdf(alpha=0.5) → 逆 CDF 插值得到中位数。对异常值比均值更鲁棒
            results["median"] = dist.icdf(
                alpha=torch.tensor(0.5, device=raw_quantiles.device, dtype=raw_quantiles.dtype)
            )
        if "quantiles" in output_type:
            # 指定分位数：默认返回 9 个分位数水平（十分位数）。通过逆 CDF 插值得到精确的分位数值，输出shape(B, test_size, 9)
            if alphas is None:
                alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            results["quantiles"] = dist.icdf(
                alpha=torch.tensor(alphas, device=raw_quantiles.device, dtype=raw_quantiles.dtype)
            )
        if "raw_quantiles" in output_type:
            # 原始分位数：直接返回全部 999 个单调分位数，不做任何聚合。输出 shape (B, test_size, 999)
            results["raw_quantiles"] = raw_quantiles

        if len(output_type) == 1:
            return results[output_type[0]] # 单个请求，直接返回 Tensor

        return results # 多个请求，返回 dict[str, Tensor]
