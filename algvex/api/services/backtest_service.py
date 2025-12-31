"""
回测服务
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from ..models.backtest import Backtest, BacktestResult, BacktestStatus
from ..schemas.backtest import BacktestCreate


logger = logging.getLogger(__name__)


class BacktestService:
    """回测服务"""

    def __init__(self, db: Session):
        self.db = db

    def create_backtest(
        self,
        backtest_data: BacktestCreate,
        owner_id: int,
    ) -> Backtest:
        """
        创建回测任务

        Args:
            backtest_data: 回测配置
            owner_id: 所有者ID

        Returns:
            Backtest 模型
        """
        backtest = Backtest(
            name=backtest_data.name,
            owner_id=owner_id,
            strategy_id=backtest_data.strategy_id,
            start_date=backtest_data.start_date,
            end_date=backtest_data.end_date,
            symbols=backtest_data.symbols,
            initial_capital=backtest_data.initial_capital,
            frequency=backtest_data.frequency,
            config=backtest_data.config,
            snapshot_id=backtest_data.snapshot_id,
            status=BacktestStatus.PENDING,
        )

        self.db.add(backtest)
        self.db.commit()
        self.db.refresh(backtest)

        logger.info(f"创建回测任务: {backtest.id}")
        return backtest

    def get_backtest(self, backtest_id: int) -> Optional[Backtest]:
        """获取回测详情"""
        return self.db.query(Backtest).filter(Backtest.id == backtest_id).first()

    def get_user_backtests(
        self,
        owner_id: int,
        skip: int = 0,
        limit: int = 20,
        status: Optional[str] = None,
    ) -> List[Backtest]:
        """获取用户的回测列表"""
        query = self.db.query(Backtest).filter(Backtest.owner_id == owner_id)

        if status:
            query = query.filter(Backtest.status == status)

        return query.order_by(Backtest.created_at.desc()).offset(skip).limit(limit).all()

    def update_backtest_status(
        self,
        backtest_id: int,
        status: str,
        progress: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> Optional[Backtest]:
        """更新回测状态"""
        backtest = self.get_backtest(backtest_id)
        if not backtest:
            return None

        backtest.status = status
        if progress is not None:
            backtest.progress = progress
        if error_message is not None:
            backtest.error_message = error_message

        if status == BacktestStatus.RUNNING and not backtest.started_at:
            backtest.started_at = datetime.utcnow()
        elif status in [BacktestStatus.COMPLETED, BacktestStatus.FAILED]:
            backtest.completed_at = datetime.utcnow()

        self.db.commit()
        self.db.refresh(backtest)

        return backtest

    def save_backtest_result(
        self,
        backtest_id: int,
        result_data: Dict[str, Any],
    ) -> Optional[BacktestResult]:
        """保存回测结果"""
        backtest = self.get_backtest(backtest_id)
        if not backtest:
            return None

        result = BacktestResult(
            backtest_id=backtest_id,
            total_return=result_data.get("total_return"),
            annual_return=result_data.get("annual_return"),
            sharpe_ratio=result_data.get("sharpe_ratio"),
            sortino_ratio=result_data.get("sortino_ratio"),
            calmar_ratio=result_data.get("calmar_ratio"),
            max_drawdown=result_data.get("max_drawdown"),
            max_drawdown_duration=result_data.get("max_drawdown_duration"),
            volatility=result_data.get("volatility"),
            total_trades=result_data.get("total_trades"),
            win_rate=result_data.get("win_rate"),
            profit_factor=result_data.get("profit_factor"),
            avg_trade_return=result_data.get("avg_trade_return"),
            equity_curve=result_data.get("equity_curve"),
            monthly_returns=result_data.get("monthly_returns"),
            trade_list=result_data.get("trade_list"),
        )

        self.db.add(result)
        self.db.commit()
        self.db.refresh(result)

        logger.info(f"保存回测结果: backtest_id={backtest_id}")
        return result

    def delete_backtest(self, backtest_id: int, owner_id: int) -> bool:
        """删除回测"""
        backtest = self.db.query(Backtest).filter(
            Backtest.id == backtest_id,
            Backtest.owner_id == owner_id,
        ).first()

        if not backtest:
            return False

        # 删除关联的结果
        if backtest.result:
            self.db.delete(backtest.result)

        self.db.delete(backtest)
        self.db.commit()

        return True

    def cancel_backtest(self, backtest_id: int, owner_id: int) -> bool:
        """取消回测"""
        backtest = self.db.query(Backtest).filter(
            Backtest.id == backtest_id,
            Backtest.owner_id == owner_id,
            Backtest.status.in_([BacktestStatus.PENDING, BacktestStatus.RUNNING]),
        ).first()

        if not backtest:
            return False

        backtest.status = BacktestStatus.CANCELLED
        backtest.completed_at = datetime.utcnow()
        self.db.commit()

        return True
