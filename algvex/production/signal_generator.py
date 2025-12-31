"""
AlgVex 信号生成器

功能:
- 基于因子和模型生成交易信号
- 集成风控规则
- 记录完整 Trace
- 支持多标的并行处理

使用方式:
    from production.signal_generator import SignalGenerator

    generator = SignalGenerator()
    signals = generator.generate(symbols=["BTCUSDT", "ETHUSDT"])
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .factor_engine import MVPFactorEngine, FactorValue
from .model_loader import ModelLoader, BaseModel
from ..shared.trace_logger import TraceLogger, SignalTrace
from ..shared.time_provider import TimeProvider
from ..shared.visibility_checker import VisibilityChecker


class SignalType(Enum):
    """信号类型"""
    LONG = "long"
    SHORT = "short"
    CLOSE = "close"
    HOLD = "hold"


@dataclass
class Signal:
    """交易信号"""
    symbol: str
    signal_type: SignalType
    strength: float  # [-1, 1]
    confidence: float  # [0, 1]
    timestamp: datetime
    factors: Dict[str, float]
    model_prediction: Optional[float] = None
    risk_adjusted: bool = False
    trace_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "strength": self.strength,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "factors": self.factors,
            "model_prediction": self.model_prediction,
            "risk_adjusted": self.risk_adjusted,
            "trace_id": self.trace_id,
            "metadata": self.metadata,
        }


@dataclass
class RiskConfig:
    """风控配置"""
    max_signal_strength: float = 1.0
    min_confidence: float = 0.3
    max_position_per_symbol: float = 0.1
    max_total_exposure: float = 1.0
    stop_loss_threshold: float = 0.05
    volatility_scaling: bool = True
    max_correlation: float = 0.7


class SignalGenerator:
    """信号生成器"""

    def __init__(
        self,
        factor_engine: Optional[MVPFactorEngine] = None,
        model_loader: Optional[ModelLoader] = None,
        model_path: Optional[str] = None,
        risk_config: Optional[RiskConfig] = None,
        enable_trace: bool = True,
    ):
        """
        初始化信号生成器

        Args:
            factor_engine: 因子计算引擎
            model_loader: 模型加载器
            model_path: 模型文件路径
            risk_config: 风控配置
            enable_trace: 是否启用 Trace
        """
        self.factor_engine = factor_engine or MVPFactorEngine()
        self.model_loader = model_loader or ModelLoader()
        self.risk_config = risk_config or RiskConfig()
        self.enable_trace = enable_trace

        # 加载模型
        self.model: Optional[BaseModel] = None
        if model_path:
            try:
                self.model = self.model_loader.load(model_path)
            except Exception as e:
                print(f"警告: 模型加载失败: {e}")

        # Trace 记录器
        self.trace_logger = TraceLogger() if enable_trace else None

        # 可见性检查器
        self.visibility_checker = VisibilityChecker()

    def generate(
        self,
        symbols: List[str],
        klines_data: Dict[str, pd.DataFrame],
        oi_data: Optional[Dict[str, pd.DataFrame]] = None,
        funding_data: Optional[Dict[str, pd.DataFrame]] = None,
        signal_time: Optional[datetime] = None,
    ) -> List[Signal]:
        """
        生成交易信号

        Args:
            symbols: 交易对列表
            klines_data: K线数据 {symbol: DataFrame}
            oi_data: 持仓量数据 {symbol: DataFrame}
            funding_data: 资金费率数据 {symbol: DataFrame}
            signal_time: 信号时间

        Returns:
            信号列表
        """
        if signal_time is None:
            signal_time = TimeProvider.utcnow()

        signals = []

        for symbol in symbols:
            try:
                signal = self._generate_single(
                    symbol=symbol,
                    klines=klines_data.get(symbol),
                    oi=oi_data.get(symbol) if oi_data else None,
                    funding=funding_data.get(symbol) if funding_data else None,
                    signal_time=signal_time,
                )
                if signal is not None:
                    signals.append(signal)
            except Exception as e:
                print(f"警告: {symbol} 信号生成失败: {e}")

        # 应用组合风控
        signals = self._apply_portfolio_risk(signals)

        return signals

    def _generate_single(
        self,
        symbol: str,
        klines: Optional[pd.DataFrame],
        oi: Optional[pd.DataFrame],
        funding: Optional[pd.DataFrame],
        signal_time: datetime,
    ) -> Optional[Signal]:
        """生成单个标的信号"""
        if klines is None or len(klines) < 100:
            return None

        # 计算因子
        factors = self.factor_engine.compute_all_factors(
            klines=klines,
            oi=oi,
            funding=funding,
            signal_time=signal_time,
        )

        # 提取因子值
        factor_values = {
            k: v.value for k, v in factors.items()
            if v.is_valid and not np.isnan(v.value)
        }

        if len(factor_values) < 5:  # 至少需要5个有效因子
            return None

        # 模型预测或规则信号
        if self.model is not None:
            prediction = self._model_predict(factor_values)
        else:
            prediction = self._rule_based_predict(factor_values)

        # 确定信号类型和强度
        signal_type, strength = self._determine_signal(prediction, factor_values)

        # 计算置信度
        confidence = self._compute_confidence(factors, prediction)

        # 应用风控
        strength, risk_adjusted = self._apply_risk_control(
            strength, confidence, factor_values
        )

        # 创建信号
        signal = Signal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            timestamp=signal_time,
            factors=factor_values,
            model_prediction=prediction,
            risk_adjusted=risk_adjusted,
        )

        # 记录 Trace
        if self.enable_trace and self.trace_logger:
            trace = self._create_trace(signal, factors)
            self.trace_logger.log_trace(trace)
            signal.trace_id = trace.trace_id

        return signal

    def _model_predict(self, factor_values: Dict[str, float]) -> float:
        """使用模型预测"""
        # 按模型期望的顺序排列特征
        feature_names = self.model.metadata.features
        if not feature_names:
            feature_names = sorted(factor_values.keys())

        features = np.array([
            factor_values.get(name, 0.0) for name in feature_names
        ]).reshape(1, -1)

        prediction = self.model.predict(features)[0]
        return float(prediction)

    def _rule_based_predict(self, factor_values: Dict[str, float]) -> float:
        """规则based预测（当没有模型时使用）"""
        # 简单加权组合
        weights = {
            "return_5m": 0.1,
            "return_1h": 0.15,
            "ma_cross": 0.2,
            "trend_strength": 0.15,
            "funding_momentum": 0.1,
            "oi_change_rate": 0.1,
            "vol_regime": -0.1,  # 高波动率降低信号
        }

        prediction = 0.0
        for factor, weight in weights.items():
            if factor in factor_values:
                # 标准化因子值
                value = factor_values[factor]
                normalized = np.clip(value / 0.01, -3, 3)  # 假设1%是正常波动
                prediction += weight * normalized

        return prediction

    def _determine_signal(
        self,
        prediction: float,
        factor_values: Dict[str, float],
    ) -> Tuple[SignalType, float]:
        """确定信号类型和强度"""
        # 信号强度 = tanh(prediction) 映射到 [-1, 1]
        strength = np.tanh(prediction)

        # 确定信号类型
        if abs(strength) < 0.1:
            signal_type = SignalType.HOLD
            strength = 0.0
        elif strength > 0:
            signal_type = SignalType.LONG
        else:
            signal_type = SignalType.SHORT
            strength = abs(strength)  # 做空信号强度也用正数表示

        return signal_type, strength

    def _compute_confidence(
        self,
        factors: Dict[str, FactorValue],
        prediction: float,
    ) -> float:
        """计算置信度"""
        # 基于因子一致性和有效性
        valid_count = sum(1 for f in factors.values() if f.is_valid)
        total_count = len(factors)

        factor_validity = valid_count / total_count if total_count > 0 else 0

        # 基于预测强度
        prediction_confidence = min(abs(prediction) / 2, 1.0)

        # 综合置信度
        confidence = 0.5 * factor_validity + 0.5 * prediction_confidence

        return float(confidence)

    def _apply_risk_control(
        self,
        strength: float,
        confidence: float,
        factor_values: Dict[str, float],
    ) -> Tuple[float, bool]:
        """应用风控规则"""
        risk_adjusted = False

        # 1. 置信度过低
        if confidence < self.risk_config.min_confidence:
            strength *= 0.5
            risk_adjusted = True

        # 2. 波动率调整
        if self.risk_config.volatility_scaling:
            vol_regime = factor_values.get("vol_regime", 1.0)
            if vol_regime > 1.5:  # 高波动率环境
                strength *= 0.7
                risk_adjusted = True

        # 3. 限制最大强度
        strength = min(strength, self.risk_config.max_signal_strength)

        return strength, risk_adjusted

    def _apply_portfolio_risk(self, signals: List[Signal]) -> List[Signal]:
        """应用组合层面风控"""
        if not signals:
            return signals

        # 按强度排序
        signals = sorted(signals, key=lambda s: s.strength, reverse=True)

        # 限制总敞口
        total_exposure = sum(s.strength for s in signals)
        max_exposure = self.risk_config.max_total_exposure

        if total_exposure > max_exposure:
            scale = max_exposure / total_exposure
            for signal in signals:
                signal.strength *= scale
                signal.risk_adjusted = True

        return signals

    def _create_trace(
        self,
        signal: Signal,
        factors: Dict[str, FactorValue],
    ) -> SignalTrace:
        """创建 Trace 记录"""
        factor_traces = []
        for factor_id, factor_value in factors.items():
            factor_traces.append({
                "factor_id": factor_id,
                "value": factor_value.value if factor_value.is_valid else None,
                "data_dependencies": {
                    dep: str(factor_value.data_time)
                    for dep in self.factor_engine.get_factor_info(factor_id).get("data_dependency", [])
                },
                "visible_time": str(factor_value.visible_time),
            })

        trace = self.trace_logger.create_trace(
            symbol=signal.symbol,
            signal_type=signal.signal_type.value,
            signal_strength=signal.strength,
            signal_time=signal.timestamp,
            factors=factor_traces,
            model_id=self.model.metadata.model_id if self.model else None,
            model_version=self.model.metadata.version if self.model else None,
            metadata={
                "confidence": signal.confidence,
                "risk_adjusted": signal.risk_adjusted,
            },
        )

        return trace


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    dates = pd.date_range("2024-01-01", periods=1000, freq="5min")
    np.random.seed(42)

    def create_test_klines():
        return pd.DataFrame(
            {
                "open": 100 + np.cumsum(np.random.randn(1000) * 0.1),
                "high": 101 + np.cumsum(np.random.randn(1000) * 0.1),
                "low": 99 + np.cumsum(np.random.randn(1000) * 0.1),
                "close": 100.5 + np.cumsum(np.random.randn(1000) * 0.1),
                "volume": 1000 + np.random.randint(0, 500, 1000),
            },
            index=dates,
        )

    klines_data = {
        "BTCUSDT": create_test_klines(),
        "ETHUSDT": create_test_klines(),
    }

    # 创建信号生成器
    generator = SignalGenerator(enable_trace=False)

    # 生成信号
    print("=== 信号生成 ===")
    signals = generator.generate(
        symbols=["BTCUSDT", "ETHUSDT"],
        klines_data=klines_data,
        signal_time=datetime(2024, 1, 4, 10, 5, 0),
    )

    for signal in signals:
        print(f"\n{signal.symbol}:")
        print(f"  类型: {signal.signal_type.value}")
        print(f"  强度: {signal.strength:.4f}")
        print(f"  置信度: {signal.confidence:.4f}")
        print(f"  风控调整: {signal.risk_adjusted}")
        print(f"  因子数: {len(signal.factors)}")
