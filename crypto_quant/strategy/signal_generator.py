"""
信号生成器模块

职责:
1. 将模型预测转换为交易信号
2. 信号过滤和确认
3. 仓位建议
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import numpy as np
import pandas as pd
from loguru import logger


class SignalType(Enum):
    """信号类型"""
    LONG = 1       # 做多
    SHORT = -1     # 做空
    CLOSE = 0      # 平仓
    HOLD = 2       # 持有


@dataclass
class TradingSignal:
    """交易信号"""
    symbol: str
    signal_type: SignalType
    strength: float         # 信号强度 [0, 1]
    target_position: float  # 目标仓位 [-1, 1]
    confidence: float       # 置信度 [0, 1]
    timestamp: datetime
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SignalGenerator:
    """
    信号生成器

    将模型预测转换为交易信号
    """

    def __init__(
        self,
        long_threshold: float = 0.6,
        short_threshold: float = 0.4,
        close_threshold: float = 0.1,
        confirmation_bars: int = 1,
        use_filters: bool = True,
    ):
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.close_threshold = close_threshold
        self.confirmation_bars = confirmation_bars
        self.use_filters = use_filters

        # 信号历史
        self.signal_history: Dict[str, List[TradingSignal]] = {}

        # 过滤器状态
        self.filter_states: Dict[str, dict] = {}

    def generate_signal(
        self,
        symbol: str,
        probability: float,
        current_position: float = 0,
        market_data: dict = None,
        timestamp: datetime = None,
    ) -> TradingSignal:
        """
        生成交易信号

        Args:
            symbol: 标的
            probability: 上涨概率 [0, 1]
            current_position: 当前仓位 [-1, 1]
            market_data: 市场数据 (用于过滤)
            timestamp: 时间戳

        Returns:
            交易信号
        """
        if timestamp is None:
            timestamp = datetime.now()

        if market_data is None:
            market_data = {}

        # 基础信号判断
        signal_type, strength = self._determine_signal(probability, current_position)

        # 应用过滤器
        if self.use_filters and market_data:
            signal_type, strength = self._apply_filters(
                symbol, signal_type, strength, market_data
            )

        # 计算目标仓位
        target_position = self._calculate_target_position(signal_type, strength)

        # 计算置信度
        confidence = self._calculate_confidence(probability, market_data)

        # 创建信号
        signal = TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            target_position=target_position,
            confidence=confidence,
            timestamp=timestamp,
            metadata={
                "probability": probability,
                "current_position": current_position,
                **market_data,
            },
        )

        # 记录历史
        if symbol not in self.signal_history:
            self.signal_history[symbol] = []
        self.signal_history[symbol].append(signal)

        return signal

    def _determine_signal(
        self,
        probability: float,
        current_position: float,
    ) -> Tuple[SignalType, float]:
        """判断信号类型和强度"""
        # 强多头信号
        if probability > self.long_threshold:
            strength = (probability - self.long_threshold) / (1 - self.long_threshold)
            if current_position < 0:
                return SignalType.CLOSE, strength  # 先平空
            return SignalType.LONG, strength

        # 强空头信号
        if probability < self.short_threshold:
            strength = (self.short_threshold - probability) / self.short_threshold
            if current_position > 0:
                return SignalType.CLOSE, strength  # 先平多
            return SignalType.SHORT, strength

        # 中性区域
        if abs(probability - 0.5) < self.close_threshold:
            # 有仓位则平仓
            if abs(current_position) > 0.01:
                return SignalType.CLOSE, 0.5
            return SignalType.HOLD, 0

        return SignalType.HOLD, 0

    def _apply_filters(
        self,
        symbol: str,
        signal_type: SignalType,
        strength: float,
        market_data: dict,
    ) -> Tuple[SignalType, float]:
        """应用信号过滤器"""
        # 获取/初始化过滤器状态
        if symbol not in self.filter_states:
            self.filter_states[symbol] = {"consecutive_signals": 0, "last_signal": None}

        state = self.filter_states[symbol]

        # 1. 资金费率过滤
        funding_rate = market_data.get("funding_rate", 0)
        if abs(funding_rate) > 0.001:  # 资金费率 > 0.1%
            if funding_rate > 0 and signal_type == SignalType.LONG:
                strength *= 0.7  # 降低做多信号强度
                logger.debug(f"{symbol}: Reduced long strength due to high funding rate")
            elif funding_rate < 0 and signal_type == SignalType.SHORT:
                strength *= 0.7  # 降低做空信号强度

        # 2. 波动率过滤
        volatility = market_data.get("volatility", 0)
        if volatility > 0.05:  # 高波动
            strength *= 0.8
            logger.debug(f"{symbol}: Reduced strength due to high volatility")

        # 3. 多空比过滤
        long_short_ratio = market_data.get("long_short_ratio", 1)
        if long_short_ratio > 2 and signal_type == SignalType.LONG:
            strength *= 0.8  # 多头拥挤
        elif long_short_ratio < 0.5 and signal_type == SignalType.SHORT:
            strength *= 0.8  # 空头拥挤

        # 4. 信号确认
        if state["last_signal"] == signal_type:
            state["consecutive_signals"] += 1
        else:
            state["consecutive_signals"] = 1
            state["last_signal"] = signal_type

        if state["consecutive_signals"] < self.confirmation_bars:
            # 信号未确认
            if signal_type in [SignalType.LONG, SignalType.SHORT]:
                signal_type = SignalType.HOLD
                strength = 0
                logger.debug(f"{symbol}: Signal not confirmed, waiting...")

        return signal_type, strength

    def _calculate_target_position(
        self,
        signal_type: SignalType,
        strength: float,
    ) -> float:
        """计算目标仓位"""
        if signal_type == SignalType.LONG:
            return min(strength, 1.0)
        elif signal_type == SignalType.SHORT:
            return -min(strength, 1.0)
        elif signal_type == SignalType.CLOSE:
            return 0
        return 0  # HOLD

    def _calculate_confidence(
        self,
        probability: float,
        market_data: dict,
    ) -> float:
        """计算信号置信度"""
        # 基础置信度 = 预测概率偏离0.5的程度
        base_confidence = abs(probability - 0.5) * 2

        # 市场条件调整
        adjustments = 1.0

        # 成交量确认
        volume_ratio = market_data.get("volume_ratio", 1)
        if volume_ratio > 1.5:
            adjustments *= 1.1
        elif volume_ratio < 0.5:
            adjustments *= 0.9

        # 趋势确认
        trend = market_data.get("trend", 0)
        if trend > 0 and probability > 0.5:
            adjustments *= 1.1  # 趋势一致
        elif trend < 0 and probability < 0.5:
            adjustments *= 1.1

        return min(base_confidence * adjustments, 1.0)

    def generate_batch_signals(
        self,
        predictions: pd.DataFrame,
        current_positions: Dict[str, float] = None,
        market_data: Dict[str, dict] = None,
    ) -> Dict[str, TradingSignal]:
        """
        批量生成信号

        Args:
            predictions: 预测结果 DataFrame (index=symbol, columns=[probability])
            current_positions: 当前仓位
            market_data: 市场数据

        Returns:
            信号字典
        """
        if current_positions is None:
            current_positions = {}
        if market_data is None:
            market_data = {}

        signals = {}
        for symbol in predictions.index:
            probability = predictions.loc[symbol, "probability"]
            position = current_positions.get(symbol, 0)
            mkt_data = market_data.get(symbol, {})

            signals[symbol] = self.generate_signal(
                symbol=symbol,
                probability=probability,
                current_position=position,
                market_data=mkt_data,
            )

        return signals

    def to_dataframe(self, signals: Dict[str, TradingSignal]) -> pd.DataFrame:
        """将信号转换为 DataFrame"""
        data = []
        for symbol, signal in signals.items():
            data.append({
                "symbol": symbol,
                "signal": signal.signal_type.name,
                "target_position": signal.target_position,
                "strength": signal.strength,
                "confidence": signal.confidence,
                "timestamp": signal.timestamp,
            })

        return pd.DataFrame(data)

    def get_signal_history(self, symbol: str, limit: int = 100) -> List[TradingSignal]:
        """获取信号历史"""
        history = self.signal_history.get(symbol, [])
        return history[-limit:]

    def analyze_signals(self, symbol: str) -> Dict:
        """分析信号统计"""
        history = self.signal_history.get(symbol, [])
        if not history:
            return {}

        # 统计各类信号数量
        signal_counts = {}
        for signal in history:
            name = signal.signal_type.name
            signal_counts[name] = signal_counts.get(name, 0) + 1

        # 平均置信度
        avg_confidence = np.mean([s.confidence for s in history])

        # 信号变化频率
        changes = 0
        for i in range(1, len(history)):
            if history[i].signal_type != history[i-1].signal_type:
                changes += 1

        return {
            "total_signals": len(history),
            "signal_counts": signal_counts,
            "avg_confidence": avg_confidence,
            "change_rate": changes / len(history) if history else 0,
        }


class MultiSymbolSignalGenerator:
    """
    多标的信号生成器

    管理多个标的的信号生成和组合
    """

    def __init__(
        self,
        symbols: List[str],
        max_positions: int = 5,
        **kwargs,
    ):
        self.symbols = symbols
        self.max_positions = max_positions

        # 每个标的一个信号生成器
        self.generators: Dict[str, SignalGenerator] = {
            symbol: SignalGenerator(**kwargs) for symbol in symbols
        }

    def generate_all_signals(
        self,
        predictions: Dict[str, float],
        current_positions: Dict[str, float] = None,
        market_data: Dict[str, dict] = None,
    ) -> Dict[str, TradingSignal]:
        """生成所有标的的信号"""
        if current_positions is None:
            current_positions = {}
        if market_data is None:
            market_data = {}

        signals = {}
        for symbol in self.symbols:
            if symbol in predictions:
                signals[symbol] = self.generators[symbol].generate_signal(
                    symbol=symbol,
                    probability=predictions[symbol],
                    current_position=current_positions.get(symbol, 0),
                    market_data=market_data.get(symbol, {}),
                )

        return signals

    def rank_signals(
        self,
        signals: Dict[str, TradingSignal],
    ) -> List[Tuple[str, TradingSignal]]:
        """
        根据信号强度和置信度排序

        Returns:
            排序后的 (symbol, signal) 列表
        """
        # 计算综合得分
        def score(signal: TradingSignal) -> float:
            if signal.signal_type == SignalType.HOLD:
                return -float('inf')
            return signal.strength * signal.confidence

        ranked = sorted(
            signals.items(),
            key=lambda x: score(x[1]),
            reverse=True,
        )

        return ranked

    def select_top_signals(
        self,
        signals: Dict[str, TradingSignal],
        n: int = None,
    ) -> Dict[str, TradingSignal]:
        """
        选择最强的N个信号

        Args:
            signals: 所有信号
            n: 选择数量，默认为 max_positions

        Returns:
            选中的信号
        """
        if n is None:
            n = self.max_positions

        ranked = self.rank_signals(signals)
        selected = {}

        for symbol, signal in ranked[:n]:
            if signal.signal_type != SignalType.HOLD:
                selected[symbol] = signal

        return selected


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 创建信号生成器
    generator = SignalGenerator(
        long_threshold=0.6,
        short_threshold=0.4,
        confirmation_bars=1,
    )

    # 模拟预测和市场数据
    symbol = "btcusdt"
    probability = 0.75
    market_data = {
        "funding_rate": 0.0005,
        "volatility": 0.03,
        "long_short_ratio": 1.2,
        "volume_ratio": 1.5,
        "trend": 1,
    }

    # 生成信号
    signal = generator.generate_signal(
        symbol=symbol,
        probability=probability,
        current_position=0,
        market_data=market_data,
    )

    print("生成的信号:")
    print(f"  标的: {signal.symbol}")
    print(f"  类型: {signal.signal_type.name}")
    print(f"  强度: {signal.strength:.4f}")
    print(f"  目标仓位: {signal.target_position:.4f}")
    print(f"  置信度: {signal.confidence:.4f}")

    # 多标的信号生成
    symbols = ["btcusdt", "ethusdt", "bnbusdt"]
    multi_gen = MultiSymbolSignalGenerator(symbols, max_positions=2)

    predictions = {
        "btcusdt": 0.75,
        "ethusdt": 0.45,
        "bnbusdt": 0.80,
    }

    all_signals = multi_gen.generate_all_signals(predictions)
    top_signals = multi_gen.select_top_signals(all_signals)

    print("\n选中的信号:")
    for sym, sig in top_signals.items():
        print(f"  {sym}: {sig.signal_type.name}, position={sig.target_position:.2f}")
