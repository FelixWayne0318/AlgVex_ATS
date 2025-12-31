"""
AlgVex 回测指标计算

计算回测结果的各类指标:
- 收益指标: 总收益率、年化收益、夏普比率
- 风险指标: 最大回撤、波动率、VaR
- 交易指标: 胜率、盈亏比、平均持仓时间
- 永续专用: 资金费用总额、爆仓次数、保证金利用率
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import math

from .models import Trade, Position, Account


@dataclass
class BacktestResult:
    """
    回测结果

    包含完整的回测输出，包括:
    - 绩效指标
    - 权益曲线
    - 交易列表
    - 持仓历史
    """
    # 基本信息
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_days: float = 0.0

    # 收益指标
    total_return: float = 0.0           # 总收益率
    annual_return: float = 0.0          # 年化收益率
    monthly_returns: List[float] = field(default_factory=list)

    # 风险指标
    sharpe_ratio: float = 0.0           # 夏普比率
    sortino_ratio: float = 0.0          # 索提诺比率
    calmar_ratio: float = 0.0           # 卡尔玛比率
    max_drawdown: float = 0.0           # 最大回撤
    max_drawdown_duration: int = 0      # 最大回撤持续时间 (天)
    volatility: float = 0.0             # 波动率 (年化)
    var_95: float = 0.0                 # 95% VaR
    var_99: float = 0.0                 # 99% VaR

    # 交易指标
    total_trades: int = 0               # 交易次数
    winning_trades: int = 0             # 盈利交易数
    losing_trades: int = 0              # 亏损交易数
    win_rate: float = 0.0               # 胜率
    profit_factor: float = 0.0          # 盈亏比
    avg_trade_return: float = 0.0       # 平均交易收益
    avg_win: float = 0.0                # 平均盈利
    avg_loss: float = 0.0               # 平均亏损
    largest_win: float = 0.0            # 最大单笔盈利
    largest_loss: float = 0.0           # 最大单笔亏损
    avg_holding_time: float = 0.0       # 平均持仓时间 (小时)

    # 永续合约专用指标
    total_funding_paid: float = 0.0     # 资金费用总额
    funding_ratio: float = 0.0          # 资金费用占比
    liquidation_count: int = 0          # 爆仓次数
    avg_margin_usage: float = 0.0       # 平均保证金利用率
    max_leverage_used: float = 0.0      # 最大使用杠杆

    # 成本分析
    total_fees: float = 0.0             # 总手续费
    total_slippage: float = 0.0         # 总滑点成本
    total_costs: float = 0.0            # 总成本

    # 资金曲线
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    drawdown_curve: List[Tuple[datetime, float]] = field(default_factory=list)

    # 交易记录
    trades: List[Trade] = field(default_factory=list)
    positions: List[Position] = field(default_factory=list)

    # 配置信息
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            # 基本信息
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_days": self.duration_days,

            # 收益指标
            "total_return": self.total_return,
            "annual_return": self.annual_return,

            # 风险指标
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "volatility": self.volatility,

            # 交易指标
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_trade_return": self.avg_trade_return,
            "avg_holding_time": self.avg_holding_time,

            # 永续专用
            "total_funding_paid": self.total_funding_paid,
            "funding_ratio": self.funding_ratio,
            "liquidation_count": self.liquidation_count,

            # 成本
            "total_fees": self.total_fees,
            "total_slippage": self.total_slippage,
            "total_costs": self.total_costs,

            # 数量
            "num_trades": len(self.trades),
            "num_positions": len(self.positions),
        }

    def get_summary(self) -> str:
        """获取摘要文本"""
        lines = [
            "=" * 60,
            "回测结果摘要",
            "=" * 60,
            f"时间范围: {self.start_time} ~ {self.end_time}",
            f"持续天数: {self.duration_days:.1f}",
            "",
            "=== 收益指标 ===",
            f"总收益率: {self.total_return:.2%}",
            f"年化收益: {self.annual_return:.2%}",
            "",
            "=== 风险指标 ===",
            f"夏普比率: {self.sharpe_ratio:.2f}",
            f"最大回撤: {self.max_drawdown:.2%}",
            f"波动率: {self.volatility:.2%}",
            "",
            "=== 交易指标 ===",
            f"交易次数: {self.total_trades}",
            f"胜率: {self.win_rate:.2%}",
            f"盈亏比: {self.profit_factor:.2f}",
            "",
            "=== 永续合约 ===",
            f"资金费用: ${self.total_funding_paid:.2f}",
            f"爆仓次数: {self.liquidation_count}",
            "",
            "=== 成本 ===",
            f"总手续费: ${self.total_fees:.2f}",
            f"总滑点: ${self.total_slippage:.2f}",
            "=" * 60,
        ]
        return "\n".join(lines)


class BacktestMetrics:
    """
    回测指标计算器

    从交易记录和权益曲线计算各类绩效指标
    """

    # 常量
    TRADING_DAYS_PER_YEAR = 365  # 加密货币 24/7 交易
    RISK_FREE_RATE = 0.02        # 无风险利率 (2%)

    def __init__(
        self,
        risk_free_rate: float = RISK_FREE_RATE,
        trading_days: int = TRADING_DAYS_PER_YEAR,
    ):
        """
        初始化指标计算器

        Args:
            risk_free_rate: 无风险利率 (年化)
            trading_days: 年交易天数
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

    def calculate_metrics(
        self,
        account: Account,
        equity_curve: List[Tuple[datetime, float]],
        trades: List[Trade],
        positions: List[Position],
        start_time: datetime,
        end_time: datetime,
        config: Optional[Dict[str, Any]] = None,
    ) -> BacktestResult:
        """
        计算完整的回测指标

        Args:
            account: 账户状态
            equity_curve: 权益曲线 [(时间, 权益), ...]
            trades: 交易列表
            positions: 持仓列表
            start_time: 开始时间
            end_time: 结束时间
            config: 回测配置

        Returns:
            BacktestResult 对象
        """
        result = BacktestResult(
            start_time=start_time,
            end_time=end_time,
            config=config or {},
            trades=trades,
            positions=positions,
            equity_curve=equity_curve,
        )

        # 计算持续时间
        duration = end_time - start_time
        result.duration_days = duration.total_seconds() / (24 * 3600)

        # 从权益曲线提取收益序列
        returns = self._calculate_returns(equity_curve)

        # 收益指标
        result.total_return = account.total_return
        result.annual_return = self._annualize_return(result.total_return, result.duration_days)

        # 风险指标
        result.volatility = self._calculate_volatility(returns)
        result.sharpe_ratio = self._calculate_sharpe(result.annual_return, result.volatility)
        result.sortino_ratio = self._calculate_sortino(returns, result.annual_return)
        result.max_drawdown, result.max_drawdown_duration = self._calculate_max_drawdown(equity_curve)
        result.calmar_ratio = self._calculate_calmar(result.annual_return, result.max_drawdown)
        result.drawdown_curve = self._calculate_drawdown_curve(equity_curve)

        if returns:
            result.var_95 = self._calculate_var(returns, 0.95)
            result.var_99 = self._calculate_var(returns, 0.99)

        # 交易指标
        trade_metrics = self._calculate_trade_metrics(trades)
        result.total_trades = trade_metrics["total_trades"]
        result.winning_trades = trade_metrics["winning_trades"]
        result.losing_trades = trade_metrics["losing_trades"]
        result.win_rate = trade_metrics["win_rate"]
        result.profit_factor = trade_metrics["profit_factor"]
        result.avg_trade_return = trade_metrics["avg_trade_return"]
        result.avg_win = trade_metrics["avg_win"]
        result.avg_loss = trade_metrics["avg_loss"]
        result.largest_win = trade_metrics["largest_win"]
        result.largest_loss = trade_metrics["largest_loss"]

        # 持仓时间
        result.avg_holding_time = self._calculate_avg_holding_time(positions)

        # 永续合约专用
        result.total_funding_paid = account.total_funding
        result.funding_ratio = (
            abs(result.total_funding_paid) / account.initial_balance
            if account.initial_balance > 0 else 0
        )
        result.liquidation_count = sum(1 for p in positions if p.liquidated)
        result.avg_margin_usage = self._calculate_avg_margin_usage(equity_curve, account)
        result.max_leverage_used = max((p.leverage for p in positions), default=1.0)

        # 成本
        result.total_fees = account.total_fees
        result.total_slippage = account.total_slippage
        result.total_costs = result.total_fees + result.total_slippage + abs(result.total_funding_paid)

        return result

    def _calculate_returns(
        self,
        equity_curve: List[Tuple[datetime, float]],
    ) -> List[float]:
        """计算收益率序列"""
        if len(equity_curve) < 2:
            return []

        returns = []
        for i in range(1, len(equity_curve)):
            prev_equity = equity_curve[i - 1][1]
            curr_equity = equity_curve[i][1]
            if prev_equity > 0:
                ret = (curr_equity - prev_equity) / prev_equity
                returns.append(ret)

        return returns

    def _annualize_return(self, total_return: float, days: float) -> float:
        """年化收益率"""
        if days <= 0:
            return 0.0
        return (1 + total_return) ** (self.trading_days / days) - 1

    def _calculate_volatility(self, returns: List[float]) -> float:
        """计算年化波动率"""
        if len(returns) < 2:
            return 0.0

        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        daily_vol = math.sqrt(variance)

        # 年化
        return daily_vol * math.sqrt(self.trading_days)

    def _calculate_sharpe(self, annual_return: float, volatility: float) -> float:
        """计算夏普比率"""
        if volatility == 0:
            return 0.0
        return (annual_return - self.risk_free_rate) / volatility

    def _calculate_sortino(self, returns: List[float], annual_return: float) -> float:
        """计算索提诺比率 (只考虑下行风险)"""
        if not returns:
            return 0.0

        # 计算下行偏差
        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return float('inf') if annual_return > self.risk_free_rate else 0.0

        mean_neg = sum(r ** 2 for r in negative_returns) / len(negative_returns)
        downside_deviation = math.sqrt(mean_neg) * math.sqrt(self.trading_days)

        if downside_deviation == 0:
            return 0.0

        return (annual_return - self.risk_free_rate) / downside_deviation

    def _calculate_max_drawdown(
        self,
        equity_curve: List[Tuple[datetime, float]],
    ) -> Tuple[float, int]:
        """计算最大回撤和持续时间"""
        if not equity_curve:
            return 0.0, 0

        peak = equity_curve[0][1]
        max_dd = 0.0
        max_dd_duration = 0
        current_dd_start = None
        peak_time = equity_curve[0][0]

        for time, equity in equity_curve:
            if equity > peak:
                peak = equity
                peak_time = time
                current_dd_start = None
            else:
                if peak > 0:
                    dd = (peak - equity) / peak
                    if dd > max_dd:
                        max_dd = dd
                        if current_dd_start is None:
                            current_dd_start = peak_time
                        dd_duration = (time - current_dd_start).days
                        if dd_duration > max_dd_duration:
                            max_dd_duration = dd_duration

        return max_dd, max_dd_duration

    def _calculate_drawdown_curve(
        self,
        equity_curve: List[Tuple[datetime, float]],
    ) -> List[Tuple[datetime, float]]:
        """计算回撤曲线"""
        if not equity_curve:
            return []

        peak = equity_curve[0][1]
        drawdown_curve = []

        for time, equity in equity_curve:
            if equity > peak:
                peak = equity

            dd = (peak - equity) / peak if peak > 0 else 0
            drawdown_curve.append((time, dd))

        return drawdown_curve

    def _calculate_calmar(self, annual_return: float, max_drawdown: float) -> float:
        """计算卡尔玛比率"""
        if max_drawdown == 0:
            return 0.0
        return annual_return / max_drawdown

    def _calculate_var(self, returns: List[float], confidence: float) -> float:
        """计算 VaR (Value at Risk)"""
        if not returns:
            return 0.0

        sorted_returns = sorted(returns)
        index = int((1 - confidence) * len(sorted_returns))
        return -sorted_returns[index] if index < len(sorted_returns) else 0.0

    def _calculate_trade_metrics(self, trades: List[Trade]) -> Dict[str, Any]:
        """计算交易指标"""
        # 只统计平仓交易
        closed_trades = [t for t in trades if t.is_close]

        if not closed_trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_trade_return": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
            }

        pnls = [t.pnl for t in closed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        total_win = sum(wins) if wins else 0
        total_loss = abs(sum(losses)) if losses else 0

        return {
            "total_trades": len(closed_trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(closed_trades) if closed_trades else 0,
            "profit_factor": total_win / total_loss if total_loss > 0 else float('inf'),
            "avg_trade_return": sum(pnls) / len(pnls) if pnls else 0,
            "avg_win": sum(wins) / len(wins) if wins else 0,
            "avg_loss": sum(losses) / len(losses) if losses else 0,
            "largest_win": max(wins) if wins else 0,
            "largest_loss": min(losses) if losses else 0,
        }

    def _calculate_avg_holding_time(self, positions: List[Position]) -> float:
        """计算平均持仓时间 (小时)"""
        holding_times = []

        for pos in positions:
            if pos.entry_time and pos.last_update_time:
                duration = (pos.last_update_time - pos.entry_time).total_seconds() / 3600
                holding_times.append(duration)

        return sum(holding_times) / len(holding_times) if holding_times else 0

    def _calculate_avg_margin_usage(
        self,
        equity_curve: List[Tuple[datetime, float]],
        account: Account,
    ) -> float:
        """计算平均保证金利用率"""
        # 简化实现：使用最终状态
        if account.equity == 0:
            return 0.0
        return account.used_margin / account.equity

    def calculate_monthly_returns(
        self,
        equity_curve: List[Tuple[datetime, float]],
    ) -> List[Dict[str, Any]]:
        """
        计算月度收益

        Returns:
            月度收益列表 [{"year": 2024, "month": 1, "return": 0.05}, ...]
        """
        if len(equity_curve) < 2:
            return []

        monthly = {}

        for i, (time, equity) in enumerate(equity_curve):
            key = (time.year, time.month)
            if key not in monthly:
                monthly[key] = {"start": equity, "end": equity}
            monthly[key]["end"] = equity

        results = []
        for (year, month), values in sorted(monthly.items()):
            if values["start"] > 0:
                ret = (values["end"] - values["start"]) / values["start"]
                results.append({
                    "year": year,
                    "month": month,
                    "return": ret,
                })

        return results
