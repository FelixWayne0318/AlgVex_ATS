"""
信号服务
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from ..models.signal import Signal, SignalTrace
from ..schemas.signal import SignalCreate


logger = logging.getLogger(__name__)


class SignalService:
    """信号服务"""

    def __init__(self, db: Session):
        self.db = db

    def create_signal(self, signal_data: SignalCreate) -> Signal:
        """
        创建信号

        Args:
            signal_data: 信号数据

        Returns:
            Signal 模型
        """
        signal_id = f"sig_{signal_data.symbol}_{signal_data.bar_time.strftime('%Y%m%d%H%M')}_{uuid.uuid4().hex[:8]}"

        signal = Signal(
            signal_id=signal_id,
            symbol=signal_data.symbol,
            frequency=signal_data.frequency,
            bar_time=signal_data.bar_time,
            raw_prediction=signal_data.raw_prediction,
            final_signal=signal_data.final_signal,
            factors=signal_data.factors,
            data_hash=signal_data.data_hash,
            features_hash=signal_data.features_hash,
            config_hash=signal_data.config_hash,
            mode=signal_data.mode,
        )

        self.db.add(signal)
        self.db.commit()
        self.db.refresh(signal)

        return signal

    def get_signal(self, signal_id: str) -> Optional[Signal]:
        """获取信号详情"""
        return self.db.query(Signal).filter(Signal.signal_id == signal_id).first()

    def get_signals(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        mode: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Signal]:
        """获取信号列表"""
        query = self.db.query(Signal)

        if symbol:
            query = query.filter(Signal.symbol == symbol)
        if start_time:
            query = query.filter(Signal.bar_time >= start_time)
        if end_time:
            query = query.filter(Signal.bar_time <= end_time)
        if mode:
            query = query.filter(Signal.mode == mode)

        return query.order_by(Signal.bar_time.desc()).offset(skip).limit(limit).all()

    def get_latest_signals(
        self,
        symbols: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[Signal]:
        """获取最新信号"""
        query = self.db.query(Signal)

        if symbols:
            query = query.filter(Signal.symbol.in_(symbols))

        return query.order_by(Signal.created_at.desc()).limit(limit).all()

    def add_trace(
        self,
        signal_id: int,
        trace_type: str,
        step_name: str,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
    ) -> SignalTrace:
        """添加信号追踪"""
        trace = SignalTrace(
            signal_id=signal_id,
            trace_type=trace_type,
            step_name=step_name,
            input_data=input_data,
            output_data=output_data,
            metadata=metadata,
            duration_ms=duration_ms,
        )

        self.db.add(trace)
        self.db.commit()
        self.db.refresh(trace)

        return trace

    def get_signal_traces(self, signal_id: int) -> List[SignalTrace]:
        """获取信号的所有追踪记录"""
        return self.db.query(SignalTrace).filter(
            SignalTrace.signal_id == signal_id
        ).order_by(SignalTrace.created_at).all()

    def get_signal_summary(self, date: str) -> Dict[str, Any]:
        """
        获取某日的信号摘要

        Args:
            date: 日期字符串 YYYY-MM-DD

        Returns:
            信号摘要
        """
        start_time = datetime.strptime(date, "%Y-%m-%d")
        end_time = start_time + timedelta(days=1)

        signals = self.db.query(Signal).filter(
            Signal.bar_time >= start_time,
            Signal.bar_time < end_time,
        ).all()

        buy_count = sum(1 for s in signals if s.final_signal == 1)
        sell_count = sum(1 for s in signals if s.final_signal == -1)
        neutral_count = sum(1 for s in signals if s.final_signal == 0)
        symbols = list(set(s.symbol for s in signals))

        return {
            "date": date,
            "total_signals": len(signals),
            "buy_signals": buy_count,
            "sell_signals": sell_count,
            "neutral_signals": neutral_count,
            "symbols": symbols,
        }

    def delete_old_signals(self, days: int = 90) -> int:
        """
        删除旧信号

        Args:
            days: 保留天数

        Returns:
            删除的信号数量
        """
        cutoff_time = datetime.utcnow() - timedelta(days=days)

        # 先删除相关的追踪记录
        old_signals = self.db.query(Signal).filter(
            Signal.created_at < cutoff_time
        ).all()

        signal_ids = [s.id for s in old_signals]

        if signal_ids:
            self.db.query(SignalTrace).filter(
                SignalTrace.signal_id.in_(signal_ids)
            ).delete(synchronize_session=False)

            deleted_count = self.db.query(Signal).filter(
                Signal.id.in_(signal_ids)
            ).delete(synchronize_session=False)

            self.db.commit()
            logger.info(f"删除了 {deleted_count} 条旧信号")
            return deleted_count

        return 0
