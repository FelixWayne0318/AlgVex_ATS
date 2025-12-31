"""
信号到交易适配器 - 连接Qlib信号和ValueCell执行层

职责:
1. 将Qlib模型信号转换为交易指令
2. 计算目标仓位和交易量
3. 风控检查
4. 调用ValueCell执行
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from loguru import logger


class TradeAction(Enum):
    """交易动作"""
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    NOOP = "noop"


@dataclass
class TradeInstruction:
    """交易指令"""
    symbol: str
    action: TradeAction
    quantity: float
    leverage: float = 1.0
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    meta: Optional[Dict] = None


@dataclass
class ExecutionResult:
    """执行结果"""
    symbol: str
    action: TradeAction
    requested_qty: float
    filled_qty: float
    avg_price: float
    fee: float
    status: str  # "filled", "partial", "rejected", "error"
    message: str = ""


class SignalToTradeAdapter:
    """
    信号到交易适配器

    将Qlib生成的连续信号转换为离散的交易指令
    """

    def __init__(
        self,
        capital: float = 100000.0,
        max_position_pct: float = 0.2,      # 单个标的最大仓位占比
        leverage: float = 1.0,               # 默认杠杆
        signal_threshold: float = 0.3,       # 信号阈值
        min_trade_value: float = 100.0,      # 最小交易金额
    ):
        self.capital = capital
        self.max_position_pct = max_position_pct
        self.leverage = leverage
        self.signal_threshold = signal_threshold
        self.min_trade_value = min_trade_value

        # 当前持仓状态
        self.positions: Dict[str, float] = {}  # symbol -> amount (正=多, 负=空)
        self.position_values: Dict[str, float] = {}  # symbol -> value

    def update_positions(self, positions: Dict[str, float], prices: Dict[str, float]):
        """更新持仓状态"""
        self.positions = positions.copy()
        self.position_values = {
            symbol: abs(amount) * prices.get(symbol, 0)
            for symbol, amount in positions.items()
        }

    def signal_to_target_position(
        self,
        symbol: str,
        signal: float,
        price: float,
    ) -> float:
        """
        将信号转换为目标持仓量

        Args:
            symbol: 交易对
            signal: 信号值 (-1 到 1)
            price: 当前价格

        Returns:
            目标持仓量 (正=多, 负=空)
        """
        # 最大仓位价值
        max_position_value = self.capital * self.max_position_pct * self.leverage

        # 根据信号强度计算目标仓位
        if abs(signal) < self.signal_threshold:
            # 信号太弱，不持仓
            target_value = 0
        else:
            # 按信号强度分配仓位
            target_value = max_position_value * signal

        # 转换为数量
        target_amount = target_value / price if price > 0 else 0

        return target_amount

    def calculate_trades(
        self,
        signals: Dict[str, float],
        prices: Dict[str, float],
    ) -> List[TradeInstruction]:
        """
        计算需要执行的交易

        Args:
            signals: {symbol: signal_value}
            prices: {symbol: current_price}

        Returns:
            交易指令列表
        """
        instructions = []

        for symbol, signal in signals.items():
            if symbol not in prices:
                continue

            price = prices[symbol]
            current_pos = self.positions.get(symbol, 0)
            target_pos = self.signal_to_target_position(symbol, signal, price)

            # 计算需要的变化
            diff = target_pos - current_pos

            # 检查最小交易金额
            trade_value = abs(diff) * price
            if trade_value < self.min_trade_value:
                continue

            # 生成交易指令
            instruction = self._generate_instruction(
                symbol=symbol,
                current_pos=current_pos,
                target_pos=target_pos,
                price=price,
            )

            if instruction:
                instructions.append(instruction)

        return instructions

    def _generate_instruction(
        self,
        symbol: str,
        current_pos: float,
        target_pos: float,
        price: float,
    ) -> Optional[TradeInstruction]:
        """生成交易指令"""
        diff = target_pos - current_pos

        if abs(diff) < 1e-8:
            return None

        # 确定交易动作
        if current_pos >= 0 and target_pos > current_pos:
            # 无仓位/多头 -> 加多
            action = TradeAction.OPEN_LONG
            quantity = diff

        elif current_pos <= 0 and target_pos < current_pos:
            # 无仓位/空头 -> 加空
            action = TradeAction.OPEN_SHORT
            quantity = abs(diff)

        elif current_pos > 0 and target_pos < current_pos:
            if target_pos >= 0:
                # 减多
                action = TradeAction.CLOSE_LONG
                quantity = current_pos - target_pos
            else:
                # 平多 + 开空 (分两步)
                # 这里简化，先平多
                action = TradeAction.CLOSE_LONG
                quantity = current_pos

        elif current_pos < 0 and target_pos > current_pos:
            if target_pos <= 0:
                # 减空
                action = TradeAction.CLOSE_SHORT
                quantity = abs(current_pos) - abs(target_pos)
            else:
                # 平空 + 开多 (分两步)
                action = TradeAction.CLOSE_SHORT
                quantity = abs(current_pos)

        else:
            return None

        return TradeInstruction(
            symbol=symbol,
            action=action,
            quantity=abs(quantity),
            leverage=self.leverage,
            meta={"current_pos": current_pos, "target_pos": target_pos},
        )

    def process_signals(
        self,
        signal_df: pd.DataFrame,
        price_df: pd.DataFrame,
        timestamp: pd.Timestamp,
    ) -> List[TradeInstruction]:
        """
        处理信号DataFrame

        Args:
            signal_df: 信号数据 (index: instrument, column: signal)
            price_df: 价格数据 (index: instrument, column: $close)
            timestamp: 当前时间

        Returns:
            交易指令列表
        """
        try:
            # 提取当前时刻数据
            current_signals = signal_df.xs(timestamp, level="datetime") if "datetime" in signal_df.index.names else signal_df
            current_prices = price_df.xs(timestamp, level="datetime") if "datetime" in price_df.index.names else price_df

            signals = {}
            prices = {}

            for inst in current_signals.index:
                if "signal" in current_signals.columns:
                    signals[inst] = current_signals.loc[inst, "signal"]
                else:
                    signals[inst] = float(current_signals.loc[inst])

                if inst in current_prices.index:
                    prices[inst] = current_prices.loc[inst, "$close"]

            return self.calculate_trades(signals, prices)

        except Exception as e:
            logger.error(f"Error processing signals: {e}")
            return []


class ValueCellExecutor:
    """
    ValueCell执行器封装

    将交易指令发送到ValueCell的CCXTExecutionGateway执行
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: str = "",
        secret_key: str = "",
        testnet: bool = True,
        margin_mode: str = "cross",
    ):
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        self.margin_mode = margin_mode
        self.gateway = None

    async def initialize(self):
        """初始化执行网关"""
        # 注意: 实际使用时需要导入ValueCell
        # from valuecell.agents.common.trading.execution import CCXTExecutionGateway
        #
        # self.gateway = CCXTExecutionGateway(
        #     exchange_id=self.exchange_id,
        #     api_key=self.api_key,
        #     secret_key=self.secret_key,
        #     testnet=self.testnet,
        #     default_type="swap",
        #     margin_mode=self.margin_mode,
        # )
        logger.info(f"ValueCell executor initialized for {self.exchange_id}")

    async def execute(self, instructions: List[TradeInstruction]) -> List[ExecutionResult]:
        """
        执行交易指令

        Args:
            instructions: 交易指令列表

        Returns:
            执行结果列表
        """
        results = []

        for inst in instructions:
            try:
                # 实际执行时调用ValueCell
                # result = await self.gateway.execute([...])

                # 模拟执行
                result = ExecutionResult(
                    symbol=inst.symbol,
                    action=inst.action,
                    requested_qty=inst.quantity,
                    filled_qty=inst.quantity,  # 假设全部成交
                    avg_price=0.0,  # 需要从实际执行结果获取
                    fee=0.0,
                    status="filled",
                    message="simulated",
                )
                results.append(result)

                logger.info(f"Executed: {inst.action.value} {inst.quantity} {inst.symbol}")

            except Exception as e:
                results.append(ExecutionResult(
                    symbol=inst.symbol,
                    action=inst.action,
                    requested_qty=inst.quantity,
                    filled_qty=0.0,
                    avg_price=0.0,
                    fee=0.0,
                    status="error",
                    message=str(e),
                ))
                logger.error(f"Execution failed for {inst.symbol}: {e}")

        return results

    async def close(self):
        """关闭连接"""
        if self.gateway:
            await self.gateway.close()


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 创建适配器
    adapter = SignalToTradeAdapter(
        capital=100000,
        max_position_pct=0.2,
        leverage=2.0,
        signal_threshold=0.3,
    )

    # 模拟信号和价格
    signals = {
        "btcusdt": 0.8,   # 强多信号
        "ethusdt": -0.5,  # 中等空信号
        "bnbusdt": 0.1,   # 弱信号 (不交易)
    }

    prices = {
        "btcusdt": 40000,
        "ethusdt": 2000,
        "bnbusdt": 300,
    }

    # 计算交易
    instructions = adapter.calculate_trades(signals, prices)

    print("生成的交易指令:")
    for inst in instructions:
        print(f"  {inst.action.value}: {inst.quantity:.4f} {inst.symbol}")
