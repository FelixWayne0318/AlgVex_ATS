#!/usr/bin/env python3
"""
AlgVex Paper Trading 启动脚本

启动模拟交易环境，验证执行层集成

用法:
    python paper_trading.py --config config/paper_trading.yaml
    python paper_trading.py --pairs BTCUSDT,ETHUSDT --duration 24h
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from algvex.core.execution import (
    AlgVexController,
    AlgVexControllerConfig,
    AlgVexEventHandler,
    AlgVexOrderTracker,
    OrderState,
    StateSynchronizer,
    TraceWriter,
)
from algvex.core.execution.state_synchronizer import PositionManager as StatePositionManager
from algvex.core.execution.controllers.algvex_controller import (
    MockSignalGenerator,
    Signal,
    PositionSide,
    ActionType,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("paper_trading")


class PaperTradingEngine:
    """
    Paper Trading 引擎

    模拟交易环境，验证执行层组件集成

    组件:
    - AlgVexController: 信号 → 动作
    - AlgVexOrderTracker: 订单追踪
    - AlgVexEventHandler: 事件处理
    - StateSynchronizer: 状态同步
    """

    def __init__(
        self,
        trading_pairs: Set[str],
        initial_capital: Decimal = Decimal("10000"),
        signal_interval: float = 60.0,  # 信号生成间隔（秒）
        sync_interval: float = 30.0,  # 状态同步间隔（秒）
        output_dir: str = "logs/paper_trading",
    ):
        self.trading_pairs = trading_pairs
        self.initial_capital = initial_capital
        self.signal_interval = signal_interval
        self.sync_interval = sync_interval
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化组件
        self._init_components()

        # 运行状态
        self._running = False
        self._start_time: Optional[float] = None

        # 模拟数据
        self._simulated_prices: Dict[str, Decimal] = {}
        self._simulated_positions: Dict[str, Dict] = {}
        self._order_counter = 0

        # 统计
        self._stats = {
            "signals_generated": 0,
            "orders_created": 0,
            "orders_filled": 0,
            "orders_cancelled": 0,
            "orders_failed": 0,
            "total_pnl": Decimal("0"),
            "current_capital": initial_capital,
        }

        logger.info(f"PaperTradingEngine initialized with {len(trading_pairs)} pairs")

    def _init_components(self):
        """初始化执行层组件"""
        # Trace 写入器
        trace_file = self.output_dir / f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.trace_writer = TraceWriter(output_file=str(trace_file), buffer_size=10)

        # 订单追踪器
        self.order_tracker = AlgVexOrderTracker(
            order_timeout=300.0,  # 5分钟超时
            max_history_size=10000,
            on_order_complete=self._on_order_complete,
            on_order_timeout=self._on_order_timeout,
        )

        # 仓位管理器（状态同步用）
        self.position_manager = StatePositionManager()

        # 事件处理器
        self.event_handler = AlgVexEventHandler(
            trace_writer=self.trace_writer,
            order_tracker=self.order_tracker,
            position_manager=self.position_manager,
            alert_handler=self._alert_handler,
        )

        # 状态同步器
        self.state_synchronizer = StateSynchronizer(
            position_manager=self.position_manager,
            sync_interval=self.sync_interval,
            on_sync_complete=self._on_sync_complete,
            on_mismatch_detected=self._on_mismatch_detected,
        )
        self.state_synchronizer.set_exchange_fetcher(self._fetch_exchange_positions)

        # 信号生成器（模拟）
        self.signal_generator = MockSignalGenerator()

        # 控制器
        controller_config = AlgVexControllerConfig(
            trading_pairs=self.trading_pairs,
            signal_threshold=0.3,
            max_position_per_pair=Decimal("0.1"),
            leverage=1,
            stop_loss_pct=Decimal("0.03"),
            take_profit_pct=Decimal("0.06"),
            cooldown_seconds=30,
        )
        self.controller = AlgVexController(
            config=controller_config,
            signal_generator=self.signal_generator,
        )

        # 初始化模拟价格
        for pair in self.trading_pairs:
            if "BTC" in pair:
                self._simulated_prices[pair] = Decimal("50000")
            elif "ETH" in pair:
                self._simulated_prices[pair] = Decimal("3000")
            else:
                self._simulated_prices[pair] = Decimal("100")

    async def start(self, duration: Optional[timedelta] = None):
        """
        启动 Paper Trading

        Args:
            duration: 运行时长（None 表示无限运行）
        """
        self._running = True
        self._start_time = time.time()

        end_time = None
        if duration:
            end_time = self._start_time + duration.total_seconds()
            logger.info(f"Starting paper trading for {duration}")
        else:
            logger.info("Starting paper trading (indefinite)")

        # 写入启动 trace
        self.trace_writer.write({
            "type": "paper_trading_started",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trading_pairs": list(self.trading_pairs),
            "initial_capital": str(self.initial_capital),
        })

        try:
            # 启动后台任务
            tasks = [
                asyncio.create_task(self._signal_loop()),
                asyncio.create_task(self._simulation_loop()),
                asyncio.create_task(self.state_synchronizer.start()),
                asyncio.create_task(self._status_loop()),
            ]

            # 等待结束条件
            while self._running:
                if end_time and time.time() >= end_time:
                    logger.info("Duration reached, stopping...")
                    break
                await asyncio.sleep(1)

        finally:
            self._running = False
            # 取消所有任务
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

            # 写入结束 trace
            self.trace_writer.write({
                "type": "paper_trading_stopped",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "runtime_seconds": time.time() - self._start_time,
                "statistics": self._get_final_stats(),
            })
            self.trace_writer.flush()

            logger.info("Paper trading stopped")
            self._print_final_report()

    async def stop(self):
        """停止 Paper Trading"""
        self._running = False
        await self.state_synchronizer.stop()
        logger.info("Stop signal sent")

    async def _signal_loop(self):
        """信号生成循环"""
        import random

        while self._running:
            try:
                # 为每个交易对生成模拟信号
                for pair in self.trading_pairs:
                    # 生成随机信号 [-1, 1]
                    signal_value = random.uniform(-1, 1)

                    # 10% 概率生成强信号
                    if random.random() < 0.1:
                        signal_value = random.choice([-1, 1]) * random.uniform(0.6, 1.0)

                    self.signal_generator.set_signal(pair, signal_value)
                    self._stats["signals_generated"] += 1

                # 更新控制器数据
                await self.controller.update_processed_data()

                # 获取动作
                actions = self.controller.determine_executor_actions()

                # 执行动作
                for action in actions:
                    await self._execute_action(action)

            except Exception as e:
                logger.error(f"Error in signal loop: {e}")

            await asyncio.sleep(self.signal_interval)

    async def _simulation_loop(self):
        """模拟循环（价格更新、订单处理）"""
        import random

        while self._running:
            try:
                # 更新模拟价格（随机波动）
                for pair in self.trading_pairs:
                    current = self._simulated_prices[pair]
                    change_pct = Decimal(str(random.uniform(-0.001, 0.001)))
                    self._simulated_prices[pair] = current * (1 + change_pct)

                # 处理活跃订单
                active_orders = self.order_tracker.get_active_orders()
                for order in active_orders:
                    if order.state == OrderState.PENDING_CREATE:
                        # 模拟订单确认（80% 成功率）
                        if random.random() < 0.8:
                            await self._simulate_order_fill(order)
                        elif random.random() < 0.1:
                            # 订单失败
                            await self._simulate_order_failure(order)

                # 检查超时订单
                await self.order_tracker.check_timeout_orders()

            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")

            await asyncio.sleep(1)

    async def _status_loop(self):
        """状态打印循环"""
        while self._running:
            try:
                runtime = time.time() - self._start_time
                hours = int(runtime // 3600)
                minutes = int((runtime % 3600) // 60)

                positions = self.position_manager.get_all_positions()
                active_orders = len(self.order_tracker.get_active_orders())

                logger.info(
                    f"[{hours:02d}h{minutes:02d}m] "
                    f"Signals: {self._stats['signals_generated']} | "
                    f"Orders: {self._stats['orders_created']} | "
                    f"Filled: {self._stats['orders_filled']} | "
                    f"Positions: {len(positions)} | "
                    f"Active Orders: {active_orders}"
                )

            except Exception as e:
                logger.error(f"Error in status loop: {e}")

            await asyncio.sleep(60)

    async def _execute_action(self, action):
        """执行控制器动作"""
        from algvex.core.execution.controllers.algvex_controller import ActionType

        if action.action_type == ActionType.CREATE_POSITION:
            await self._create_order(action)
        elif action.action_type == ActionType.CLOSE_POSITION:
            await self._close_position(action)

    async def _create_order(self, action):
        """创建模拟订单"""
        self._order_counter += 1
        client_order_id = f"paper_{self._order_counter:06d}"

        config = action.position_config
        current_price = self._simulated_prices.get(config.trading_pair, Decimal("0"))

        # 追踪订单
        order = self.order_tracker.track_order(
            client_order_id=client_order_id,
            trading_pair=config.trading_pair,
            side="BUY" if config.side == PositionSide.LONG else "SELL",
            amount=config.amount,
            price=current_price,
            order_type="MARKET",
            signal_id=action.signal.signal_id if action.signal else None,
        )

        # 触发事件
        await self.event_handler.handle_order_created(
            client_order_id=client_order_id,
            trading_pair=config.trading_pair,
            side="BUY" if config.side == PositionSide.LONG else "SELL",
            amount=config.amount,
            price=current_price,
            order_type="MARKET",
        )

        self._stats["orders_created"] += 1
        logger.debug(f"Order created: {client_order_id}")

    async def _close_position(self, action):
        """平仓"""
        position = self.position_manager.get_position(action.trading_pair)
        if not position:
            return

        self._order_counter += 1
        client_order_id = f"paper_{self._order_counter:06d}"

        # 平仓方向与持仓相反
        side = "SELL" if position.side == "LONG" else "BUY"

        self.order_tracker.track_order(
            client_order_id=client_order_id,
            trading_pair=action.trading_pair,
            side=side,
            amount=position.amount,
            price=self._simulated_prices.get(action.trading_pair, Decimal("0")),
            order_type="MARKET",
        )

        await self.event_handler.handle_order_created(
            client_order_id=client_order_id,
            trading_pair=action.trading_pair,
            side=side,
            amount=position.amount,
            order_type="MARKET",
        )

        self._stats["orders_created"] += 1

    async def _simulate_order_fill(self, order):
        """模拟订单成交"""
        current_price = self._simulated_prices.get(order.trading_pair, Decimal("0"))
        fee = order.amount * current_price * Decimal("0.0004")  # 0.04% 手续费

        # 更新订单状态
        self.order_tracker.update_order(
            client_order_id=order.client_order_id,
            exchange_order_id=f"sim_{order.client_order_id}",
            state=OrderState.FILLED,
            filled_amount=order.amount,
            average_price=current_price,
            trade_fees=fee,
        )

        # 触发事件
        await self.event_handler.handle_order_filled(
            client_order_id=order.client_order_id,
            trading_pair=order.trading_pair,
            side=order.side,
            filled_amount=order.amount,
            average_price=current_price,
            fee=fee,
        )

        # 更新模拟仓位
        self._simulated_positions[order.trading_pair] = {
            "side": "LONG" if order.side == "BUY" else "SHORT",
            "amount": order.amount,
            "entry_price": current_price,
        }

        self._stats["orders_filled"] += 1

    async def _simulate_order_failure(self, order):
        """模拟订单失败"""
        await self.event_handler.handle_order_failed(
            client_order_id=order.client_order_id,
            trading_pair=order.trading_pair,
            error_message="Simulated order failure",
        )

        self._stats["orders_failed"] += 1

    async def _fetch_exchange_positions(self) -> Dict[str, Dict]:
        """获取模拟交易所仓位"""
        positions = {}
        for symbol, pos in self._simulated_positions.items():
            if pos["amount"] > 0:
                positions[symbol] = {
                    "side": pos["side"],
                    "amount": pos["amount"],
                    "entry_price": pos["entry_price"],
                    "unrealized_pnl": Decimal("0"),
                }
        return positions

    def _on_order_complete(self, order):
        """订单完成回调"""
        logger.debug(f"Order complete: {order.client_order_id}, state={order.state.value}")

    def _on_order_timeout(self, order):
        """订单超时回调"""
        logger.warning(f"Order timeout: {order.client_order_id}")
        self._stats["orders_cancelled"] += 1

    def _on_sync_complete(self, result):
        """同步完成回调"""
        if not result.all_synced:
            logger.warning(f"Sync issues: {result.mismatched} mismatched, {result.missing_local} missing local")

    def _on_mismatch_detected(self, symbol: str, details: Dict):
        """不一致检测回调"""
        logger.warning(f"Position mismatch detected for {symbol}: {details}")

    def _alert_handler(self, alert_type: str, details: Dict):
        """告警处理"""
        logger.error(f"ALERT [{alert_type}]: {details}")

    def _get_final_stats(self) -> Dict[str, Any]:
        """获取最终统计"""
        return {
            "signals_generated": self._stats["signals_generated"],
            "orders_created": self._stats["orders_created"],
            "orders_filled": self._stats["orders_filled"],
            "orders_cancelled": self._stats["orders_cancelled"],
            "orders_failed": self._stats["orders_failed"],
            "fill_rate": self._stats["orders_filled"] / max(1, self._stats["orders_created"]),
            "final_positions": len(self._simulated_positions),
            "order_tracker_stats": self.order_tracker.get_statistics(),
            "sync_stats": self.state_synchronizer.get_statistics(),
            "event_stats": self.event_handler.get_statistics(),
        }

    def _print_final_report(self):
        """打印最终报告"""
        stats = self._get_final_stats()
        runtime = time.time() - self._start_time

        print("\n" + "=" * 60)
        print("          PAPER TRADING FINAL REPORT")
        print("=" * 60)
        print(f"Runtime:          {runtime / 3600:.2f} hours")
        print(f"Signals:          {stats['signals_generated']}")
        print(f"Orders Created:   {stats['orders_created']}")
        print(f"Orders Filled:    {stats['orders_filled']}")
        print(f"Orders Cancelled: {stats['orders_cancelled']}")
        print(f"Orders Failed:    {stats['orders_failed']}")
        print(f"Fill Rate:        {stats['fill_rate']:.1%}")
        print(f"Final Positions:  {stats['final_positions']}")
        print("=" * 60)

        # 保存报告
        report_file = self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"Report saved to: {report_file}")


def parse_duration(duration_str: str) -> timedelta:
    """解析时长字符串"""
    duration_str = duration_str.lower().strip()

    if duration_str.endswith('h'):
        return timedelta(hours=float(duration_str[:-1]))
    elif duration_str.endswith('m'):
        return timedelta(minutes=float(duration_str[:-1]))
    elif duration_str.endswith('d'):
        return timedelta(days=float(duration_str[:-1]))
    else:
        return timedelta(hours=float(duration_str))


def main():
    parser = argparse.ArgumentParser(description="AlgVex Paper Trading")
    parser.add_argument(
        "--pairs",
        type=str,
        default="BTCUSDT,ETHUSDT",
        help="Trading pairs (comma-separated)",
    )
    parser.add_argument(
        "--duration",
        type=str,
        default=None,
        help="Duration (e.g., 1h, 30m, 24h)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000,
        help="Initial capital (USD)",
    )
    parser.add_argument(
        "--signal-interval",
        type=float,
        default=60,
        help="Signal generation interval (seconds)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/paper_trading",
        help="Output directory for logs",
    )

    args = parser.parse_args()

    # 解析交易对
    trading_pairs = set(p.strip().upper() for p in args.pairs.split(","))

    # 解析时长
    duration = None
    if args.duration:
        duration = parse_duration(args.duration)

    # 创建引擎
    engine = PaperTradingEngine(
        trading_pairs=trading_pairs,
        initial_capital=Decimal(str(args.capital)),
        signal_interval=args.signal_interval,
        output_dir=args.output_dir,
    )

    # 设置信号处理
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        asyncio.get_event_loop().create_task(engine.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 运行
    logger.info(f"Paper Trading: pairs={trading_pairs}, duration={duration}")
    asyncio.run(engine.start(duration))


if __name__ == "__main__":
    main()
