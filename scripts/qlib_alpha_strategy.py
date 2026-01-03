"""
Qlib Alpha V2 策略

基于 Qlib 机器学习模型的加密货币交易策略。

使用 Hummingbot Strategy V2 框架:
- StrategyV2Base (替代 ScriptStrategyBase)
- MarketDataProvider (替代 CandlesFactory)
- PositionExecutor (替代 buy()/sell())

启动方式:
    hummingbot
    >>> start --script qlib_alpha_strategy.py --conf conf/scripts/qlib_alpha_v2.yml
"""

import logging
from decimal import Decimal
from typing import Dict, Set

from pydantic import Field

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.strategy_v2.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase
from hummingbot.strategy_v2.models.executor_actions import ExecutorAction

# 导入控制器
from controllers.qlib_alpha_controller import (
    QlibAlphaController,
    QlibAlphaControllerConfig,
)


class QlibAlphaStrategyConfig(StrategyV2ConfigBase):
    """
    V2 策略配置

    继承自 StrategyV2ConfigBase，支持:
    - 动态配置更新 (config_update_interval)
    - 多控制器管理
    """
    script_file_name: str = "qlib_alpha_strategy.py"
    controllers_config: list = Field(default=[])

    # 可以在 YAML 中直接配置控制器
    # 或者在这里定义默认值


class QlibAlphaStrategy(StrategyV2Base):
    """
    Qlib Alpha V2 策略

    V2 架构工作流程:
    1. MarketDataProvider 自动管理 K 线数据
    2. Controller 调用 Qlib 模型生成信号
    3. PositionExecutor 自动管理订单生命周期 (含三重屏障)
    """

    @classmethod
    def init_markets(cls, config: QlibAlphaStrategyConfig):
        """
        初始化市场配置 (v10.0.4)

        直接使用 YAML 中的 markets 配置，不从 controllers_config 推导。
        controllers_config 使用 List[str] 格式 (引用配置文件名)，
        不能当作 dict 使用。
        """
        # 直接使用 config.markets (YAML 已定义)
        cls.markets = config.markets if hasattr(config, 'markets') else {}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: QlibAlphaStrategyConfig):
        super().__init__(connectors, config)
        self.logger = logging.getLogger(__name__)

    def create_actions_proposal(self) -> list[ExecutorAction]:
        """
        创建执行动作提案 (V2 核心方法)

        遍历所有控制器，收集 ExecutorAction
        """
        actions = []
        for controller in self.controllers.values():
            controller_actions = controller.determine_executor_actions()
            actions.extend(controller_actions)
        return actions

    def format_status(self) -> str:
        """状态显示"""
        lines = []
        lines.append("=== Qlib Alpha V2 Strategy ===")
        lines.append(f"Controllers: {len(self.controllers)}")

        for name, controller in self.controllers.items():
            lines.append(f"\n[Controller: {name}]")
            # v10.0.4: 使用 model_loaded (QlibAlphaController 实际存在的字段)
            lines.append(f"  Model Loaded: {controller.model_loaded}")
            lines.append(f"  Model: {type(controller.model).__name__ if controller.model else 'None'}")
            active = len(controller.get_active_executors())
            lines.append(f"  Active Executors: {active}")

        return "\n".join(lines)
