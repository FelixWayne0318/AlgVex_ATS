"""
Qlib Alpha 控制器 (v9.0.0)

基于统一特征计算的交易信号控制器。

V2 架构中，Controller 负责:
1. 从 MarketDataProvider 获取数据
2. 使用统一特征模块计算特征
3. 应用相同的归一化参数
4. 生成 ExecutorAction 供策略执行

重要变更 (v9.0.0):
- 使用 unified_features.py 计算特征，与训练完全一致
- 加载训练时保存的归一化参数
- 保证特征列顺序与训练一致
"""

import pickle
import logging
from pathlib import Path
from decimal import Decimal
from typing import List, Optional, Set

import numpy as np
import pandas as pd
import lightgbm as lgb

from hummingbot.strategy_v2.controllers.controller_base import ControllerBase, ControllerConfigBase
from hummingbot.strategy_v2.executors.executor_base import ExecutorBase
from hummingbot.strategy_v2.executors.position_executor.data_types import (
    PositionExecutorConfig,
    TrailingStop,
    TripleBarrierConfig,
    TradeType,
)
from hummingbot.strategy_v2.models.executor_actions import (
    CreateExecutorAction,
    ExecutorAction,
    StopExecutorAction,
)
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo
from pydantic import Field

# 导入统一特征模块 (兼容不同运行目录)
try:
    from scripts.unified_features import (
        compute_unified_features,
        FeatureNormalizer,
        FEATURE_COLUMNS,
    )
except ImportError:
    from unified_features import (
        compute_unified_features,
        FeatureNormalizer,
        FEATURE_COLUMNS,
    )


class QlibAlphaControllerConfig(ControllerConfigBase):
    """
    Qlib Alpha 控制器配置

    使用 StrategyV2ConfigBase 风格的配置类
    """
    id: str = Field(default="qlib_alpha_btc")  # 必需字段，用于 Executor 关联
    controller_name: str = "qlib_alpha"
    controller_type: str = "directional_trading"

    # 交易配置
    connector_name: str = Field(default="binance")
    trading_pair: str = Field(default="BTC-USDT")
    order_amount_usd: Decimal = Field(default=Decimal("100"))

    # 模型配置 (v10.0.4: 4 件套)
    # 包含 lgb_model.txt + normalizer.pkl + feature_columns.pkl + metadata.json
    model_dir: str = Field(default="~/.algvex/models/qlib_alpha")

    # 信号配置
    signal_threshold: Decimal = Field(default=Decimal("0.005"))
    prediction_interval: str = Field(default="1h")
    lookback_bars: int = Field(default=100)

    # 三重屏障配置 (用于 PositionExecutor)
    stop_loss: Decimal = Field(default=Decimal("0.02"))
    take_profit: Decimal = Field(default=Decimal("0.03"))
    time_limit: int = Field(default=3600)

    # 执行配置
    cooldown_interval: int = Field(default=60)
    max_executors_per_side: int = Field(default=1)


class QlibAlphaController(ControllerBase):
    """
    Qlib Alpha 控制器 (v9.0.0)

    V2 架构中的核心组件，负责:
    1. 接收 MarketDataProvider 数据
    2. 使用统一特征模块计算特征 (与训练完全一致)
    3. 应用训练时保存的归一化参数
    4. 生成 PositionExecutor 动作
    """

    def __init__(self, config: QlibAlphaControllerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 模型和归一化器
        self.model = None
        self.normalizer = None
        self.feature_columns = None
        self.model_loaded = False
        self.last_signal_time = 0

        # 加载模型和归一化参数
        self._load_model_and_normalizer()

    def _load_model_and_normalizer(self):
        """加载模型、归一化参数和特征列顺序"""
        try:
            model_dir = Path(self.config.model_dir).expanduser()

            # 加载 LightGBM 模型
            model_file = model_dir / "lgb_model.txt"
            if model_file.exists():
                self.model = lgb.Booster(model_file=str(model_file))
                self.logger.info(f"Model loaded from {model_file}")
            else:
                self.logger.warning(f"Model file not found: {model_file}")
                return

            # 加载归一化参数
            normalizer_file = model_dir / "normalizer.pkl"
            if normalizer_file.exists():
                self.normalizer = FeatureNormalizer()
                self.normalizer.load(str(normalizer_file))
                self.logger.info(f"Normalizer loaded from {normalizer_file}")
            else:
                self.logger.warning(f"Normalizer file not found: {normalizer_file}")
                return

            # 加载特征列顺序
            columns_file = model_dir / "feature_columns.pkl"
            if columns_file.exists():
                with open(columns_file, "rb") as f:
                    self.feature_columns = pickle.load(f)
                self.logger.info(f"Feature columns loaded: {len(self.feature_columns)} features")
            else:
                # 使用默认列顺序
                self.feature_columns = FEATURE_COLUMNS
                self.logger.info("Using default feature columns")

            self.model_loaded = True

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")

    async def update_processed_data(self):
        """
        更新处理后的数据 (V2 框架回调)

        从 MarketDataProvider 获取 K 线数据
        """
        try:
            # 使用 MarketDataProvider 获取 K 线数据
            candles_df = self.market_data_provider.get_candles_df(
                connector_name=self.config.connector_name,
                trading_pair=self.config.trading_pair,
                interval=self.config.prediction_interval,
                max_records=self.config.lookback_bars,
            )
            self.processed_data["candles"] = candles_df
        except Exception as e:
            self.logger.debug(f"Failed to get candles: {e}")
            self.processed_data["candles"] = None

    def determine_executor_actions(self) -> List[ExecutorAction]:
        """
        确定执行动作 (V2 框架核心方法)

        Returns
        -------
        List[ExecutorAction]
            要执行的动作列表 (CreateExecutorAction / StopExecutorAction)
        """
        actions = []

        # 检查冷却时间
        current_time = self.market_data_provider.time()
        if current_time - self.last_signal_time < self.config.cooldown_interval:
            return actions

        # 检查是否已有活跃的 Executor
        active_executors = self.get_active_executors()
        if len(active_executors) >= self.config.max_executors_per_side:
            return actions

        # 获取信号
        signal = self._get_signal()

        if signal != 0:
            action = self._create_position_executor(signal)
            if action:
                actions.append(action)
                self.last_signal_time = current_time

        return actions

    # 最小 K 线数量 (60 根用于滚动窗口 + 1 根用于信号)
    MIN_BARS: int = 61

    def _get_signal(self) -> int:
        """
        获取交易信号 (v10.0.0: 使用统一特征 + 闭合 bar)

        Returns
        -------
        int
            1=买入, -1=卖出, 0=持有

        Note
        ----
        使用倒数第二根 K 线 (iloc[-2]) 确保数据已闭合，
        与离线回测完全一致。
        """
        if not self.model_loaded:
            return 0

        candles = self.processed_data.get("candles")
        if candles is None or len(candles) < self.MIN_BARS:
            return 0

        try:
            # 使用统一特征模块计算特征 (与训练完全一致)
            features = compute_unified_features(candles)
            if features is None or features.empty:
                return 0

            # 确保列顺序与训练一致
            features = features[self.feature_columns]

            # v10.0.0: 取倒数第二行 (已闭合 K 线) 并应用归一化
            # 注意: iloc[-2:-1] 返回 DataFrame，iloc[-2] 返回 Series
            latest_features = features.iloc[-2:-1]
            latest_features_norm = self.normalizer.transform(latest_features, strict=True)

            # 预测
            prediction = self.model.predict(latest_features_norm.values)[0]

            # 根据阈值生成信号
            threshold = float(self.config.signal_threshold)
            if prediction > threshold:
                self.logger.info(f"BUY signal: prediction={prediction:.6f}")
                return 1
            elif prediction < -threshold:
                self.logger.info(f"SELL signal: prediction={prediction:.6f}")
                return -1

            return 0

        except Exception as e:
            self.logger.error(f"Error getting signal: {e}")
            return 0

    def _create_position_executor(self, signal: int) -> Optional[CreateExecutorAction]:
        """
        创建 PositionExecutor 动作 (v10.0.0: Decimal 精度)

        使用 TripleBarrierConfig 配置止损/止盈/时间限制

        Note
        ----
        v10.0.0: 所有金额计算统一使用 Decimal 类型，避免浮点精度问题
        """
        try:
            # 获取当前价格
            mid_price_raw = self.market_data_provider.get_price_by_type(
                connector_name=self.config.connector_name,
                trading_pair=self.config.trading_pair,
                price_type="mid",
            )

            if mid_price_raw is None or mid_price_raw <= 0:
                return None

            # v10.0.0: 强制转换为 Decimal 类型
            mid_price = Decimal(str(mid_price_raw))
            order_amount_usd = Decimal(str(self.config.order_amount_usd))

            # 计算下单数量 (Decimal 精度)
            amount = order_amount_usd / mid_price

            # 获取交易对精度信息 (如有)
            # 注: 实际部署时应从交易所获取 step_size/min_notional
            # amount = self._quantize_amount(amount, trading_pair)

            # 三重屏障配置
            triple_barrier = TripleBarrierConfig(
                stop_loss=Decimal(str(self.config.stop_loss)),
                take_profit=Decimal(str(self.config.take_profit)),
                time_limit=self.config.time_limit,
            )

            # 创建 PositionExecutor 配置
            executor_config = PositionExecutorConfig(
                timestamp=self.market_data_provider.time(),
                connector_name=self.config.connector_name,
                trading_pair=self.config.trading_pair,
                side=TradeType.BUY if signal > 0 else TradeType.SELL,
                amount=amount,  # Decimal 类型
                triple_barrier_config=triple_barrier,
            )

            self.logger.info(
                f"Creating PositionExecutor: {executor_config.side} "
                f"{amount} @ {mid_price}"
            )

            return CreateExecutorAction(
                controller_id=self.config.id,  # 使用 id 字段而非 controller_name
                executor_config=executor_config,
            )

        except Exception as e:
            self.logger.error(f"Error creating executor: {e}")
            return None

    def get_active_executors(self) -> List[ExecutorInfo]:
        """获取活跃的 Executors (v10.0.4: 返回 ExecutorInfo 而非 ExecutorBase)"""
        return [
            executor
            for executor in self.executors_info
            if executor.is_active
        ]
