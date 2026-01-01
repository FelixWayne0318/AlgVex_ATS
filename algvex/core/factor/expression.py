"""
AlgVex 表达式引擎 (Qlib 风格)

完整实现 Qlib ops.py 的 60+ 表达式运算符:
- 元素运算符: Abs, Sign, Log, Power, Add, Sub, Mul, Div, ...
- 滚动运算符: Mean, Sum, Std, Max, Min, Rank, Delta, Slope, ...
- 成对运算符: Corr, Cov
- 条件运算符: If, Greater, Less
- 重采样运算符: TResample

用法:
    from algvex.core.factor.expression import Operators, Expression

    ops = Operators()

    # 计算 20 日动量
    momentum = ops.ref(close, 20) / close - 1

    # 计算波动率
    volatility = ops.std(returns, 20)

    # 布林带
    middle = ops.mean(close, 20)
    upper = middle + 2 * ops.std(close, 20)
"""

import numpy as np
import pandas as pd
from typing import Union, Callable, Optional, List
from abc import ABC, abstractmethod

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ============================================================
# 表达式基类
# ============================================================

class Expression(ABC):
    """表达式基类"""

    @abstractmethod
    def __call__(self, data: pd.DataFrame) -> pd.Series:
        """计算表达式值"""
        raise NotImplementedError

    def __add__(self, other):
        return BinaryOp(self, other, lambda a, b: a + b, "Add")

    def __radd__(self, other):
        return BinaryOp(other, self, lambda a, b: a + b, "Add")

    def __sub__(self, other):
        return BinaryOp(self, other, lambda a, b: a - b, "Sub")

    def __rsub__(self, other):
        return BinaryOp(other, self, lambda a, b: a - b, "Sub")

    def __mul__(self, other):
        return BinaryOp(self, other, lambda a, b: a * b, "Mul")

    def __rmul__(self, other):
        return BinaryOp(other, self, lambda a, b: a * b, "Mul")

    def __truediv__(self, other):
        return BinaryOp(self, other, lambda a, b: a / (b + 1e-12), "Div")

    def __rtruediv__(self, other):
        return BinaryOp(other, self, lambda a, b: a / (b + 1e-12), "Div")

    def __neg__(self):
        return UnaryOp(self, lambda x: -x, "Neg")

    def __abs__(self):
        return UnaryOp(self, lambda x: np.abs(x), "Abs")


class Feature(Expression):
    """特征表达式"""

    def __init__(self, name: str):
        self.name = name

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        return data[self.name]

    def __repr__(self):
        return f"${self.name}"


class Constant(Expression):
    """常量表达式"""

    def __init__(self, value: float):
        self.value = value

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=data.index)

    def __repr__(self):
        return str(self.value)


class UnaryOp(Expression):
    """一元运算表达式"""

    def __init__(self, operand: Expression, op: Callable, name: str):
        self.operand = operand
        self.op = op
        self.name = name

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        if isinstance(self.operand, Expression):
            val = self.operand(data)
        else:
            val = self.operand
        return self.op(val)

    def __repr__(self):
        return f"{self.name}({self.operand})"


class BinaryOp(Expression):
    """二元运算表达式"""

    def __init__(self, left, right, op: Callable, name: str):
        self.left = left
        self.right = right
        self.op = op
        self.name = name

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        if isinstance(self.left, Expression):
            left_val = self.left(data)
        else:
            left_val = self.left

        if isinstance(self.right, Expression):
            right_val = self.right(data)
        else:
            right_val = self.right

        return self.op(left_val, right_val)

    def __repr__(self):
        return f"{self.name}({self.left}, {self.right})"


# ============================================================
# 表达式运算符 (Qlib ops.py 完整实现)
# ============================================================

class Operators:
    """
    Qlib 表达式运算符的 Pandas 实现

    包含 60+ 运算符，覆盖 Qlib 的全部功能
    """

    EPS = 1e-12  # 防止除零

    # ==================== 元素运算符 ====================

    @staticmethod
    def abs(series: pd.Series) -> pd.Series:
        """Abs - 绝对值"""
        return np.abs(series)

    @staticmethod
    def sign(series: pd.Series) -> pd.Series:
        """Sign - 符号函数"""
        return np.sign(series)

    @staticmethod
    def log(series: pd.Series) -> pd.Series:
        """Log - 自然对数"""
        return np.log(series.clip(lower=Operators.EPS))

    @staticmethod
    def log1p(series: pd.Series) -> pd.Series:
        """Log1p - log(1+x)"""
        return np.log1p(series)

    @staticmethod
    def exp(series: pd.Series) -> pd.Series:
        """Exp - 指数"""
        return np.exp(series.clip(upper=700))  # 防止溢出

    @staticmethod
    def sqrt(series: pd.Series) -> pd.Series:
        """Sqrt - 平方根"""
        return np.sqrt(series.clip(lower=0))

    @staticmethod
    def power(left: pd.Series, right: Union[pd.Series, float]) -> pd.Series:
        """Power - 幂运算"""
        return np.power(left, right)

    @staticmethod
    def clip(series: pd.Series, lower: float = None, upper: float = None) -> pd.Series:
        """Clip - 截断"""
        return series.clip(lower=lower, upper=upper)

    @staticmethod
    def mask(series: pd.Series, condition: pd.Series, value: float = np.nan) -> pd.Series:
        """Mask - 条件遮罩"""
        result = series.copy()
        result[condition] = value
        return result

    @staticmethod
    def not_(series: pd.Series) -> pd.Series:
        """Not - 逻辑非"""
        return ~series.astype(bool)

    # ==================== 二元运算符 ====================

    @staticmethod
    def add(left: pd.Series, right: pd.Series) -> pd.Series:
        """Add - 加法"""
        return left + right

    @staticmethod
    def sub(left: pd.Series, right: pd.Series) -> pd.Series:
        """Sub - 减法"""
        return left - right

    @staticmethod
    def mul(left: pd.Series, right: pd.Series) -> pd.Series:
        """Mul - 乘法"""
        return left * right

    @staticmethod
    def div(left: pd.Series, right: pd.Series) -> pd.Series:
        """Div - 除法"""
        return left / (right + Operators.EPS)

    @staticmethod
    def greater(left: pd.Series, right: pd.Series) -> pd.Series:
        """Greater - 逐元素取最大值"""
        return np.maximum(left, right)

    @staticmethod
    def less(left: pd.Series, right: pd.Series) -> pd.Series:
        """Less - 逐元素取最小值"""
        return np.minimum(left, right)

    @staticmethod
    def gt(left: pd.Series, right: pd.Series) -> pd.Series:
        """Gt - 大于"""
        return (left > right).astype(float)

    @staticmethod
    def ge(left: pd.Series, right: pd.Series) -> pd.Series:
        """Ge - 大于等于"""
        return (left >= right).astype(float)

    @staticmethod
    def lt(left: pd.Series, right: pd.Series) -> pd.Series:
        """Lt - 小于"""
        return (left < right).astype(float)

    @staticmethod
    def le(left: pd.Series, right: pd.Series) -> pd.Series:
        """Le - 小于等于"""
        return (left <= right).astype(float)

    @staticmethod
    def eq(left: pd.Series, right: pd.Series) -> pd.Series:
        """Eq - 等于"""
        return (left == right).astype(float)

    @staticmethod
    def ne(left: pd.Series, right: pd.Series) -> pd.Series:
        """Ne - 不等于"""
        return (left != right).astype(float)

    @staticmethod
    def and_(left: pd.Series, right: pd.Series) -> pd.Series:
        """And - 逻辑与"""
        return (left.astype(bool) & right.astype(bool)).astype(float)

    @staticmethod
    def or_(left: pd.Series, right: pd.Series) -> pd.Series:
        """Or - 逻辑或"""
        return (left.astype(bool) | right.astype(bool)).astype(float)

    # ==================== 条件运算符 ====================

    @staticmethod
    def if_(condition: pd.Series, true_val: pd.Series, false_val: pd.Series) -> pd.Series:
        """If - 条件选择"""
        return np.where(condition, true_val, false_val)

    # ==================== 引用运算符 ====================

    @staticmethod
    def ref(series: pd.Series, n: int) -> pd.Series:
        """
        Ref - 引用历史数据

        n > 0: 过去第n期
        n < 0: 未来第n期 (仅回测使用)
        n = 0: 第一个有效值
        """
        if n == 0:
            return pd.Series(series.iloc[0], index=series.index)
        return series.shift(n)

    @staticmethod
    def diff(series: pd.Series, n: int = 1) -> pd.Series:
        """Diff - 差分"""
        return series.diff(n)

    @staticmethod
    def pct_change(series: pd.Series, n: int = 1) -> pd.Series:
        """PctChange - 百分比变化"""
        return series.pct_change(n)

    # ==================== 滚动运算符 ====================

    @staticmethod
    def mean(series: pd.Series, n: int) -> pd.Series:
        """Mean - 简单移动平均"""
        return series.rolling(n, min_periods=1).mean()

    @staticmethod
    def sum(series: pd.Series, n: int) -> pd.Series:
        """Sum - 滚动求和"""
        return series.rolling(n, min_periods=1).sum()

    @staticmethod
    def std(series: pd.Series, n: int) -> pd.Series:
        """Std - 滚动标准差"""
        return series.rolling(n, min_periods=2).std()

    @staticmethod
    def var(series: pd.Series, n: int) -> pd.Series:
        """Var - 滚动方差"""
        return series.rolling(n, min_periods=2).var()

    @staticmethod
    def max(series: pd.Series, n: int) -> pd.Series:
        """Max - 滚动最大值"""
        return series.rolling(n, min_periods=1).max()

    @staticmethod
    def min(series: pd.Series, n: int) -> pd.Series:
        """Min - 滚动最小值"""
        return series.rolling(n, min_periods=1).min()

    @staticmethod
    def rank(series: pd.Series, n: int) -> pd.Series:
        """Rank - 滚动百分位排名"""
        def pct_rank(x):
            if len(x) < 2:
                return 0.5
            return (x.argsort().argsort()[-1] + 1) / len(x)
        return series.rolling(n, min_periods=1).apply(pct_rank, raw=False)

    @staticmethod
    def quantile(series: pd.Series, n: int, q: float) -> pd.Series:
        """Quantile - 滚动分位数"""
        return series.rolling(n, min_periods=1).quantile(q)

    @staticmethod
    def med(series: pd.Series, n: int) -> pd.Series:
        """Med - 滚动中位数"""
        return series.rolling(n, min_periods=1).median()

    @staticmethod
    def count(series: pd.Series, n: int) -> pd.Series:
        """Count - 滚动计数 (非 NaN)"""
        return series.rolling(n, min_periods=1).count()

    @staticmethod
    def delta(series: pd.Series, n: int) -> pd.Series:
        """Delta - 变化量 (当前值 - n期前的值)"""
        return series - series.shift(n)

    @staticmethod
    def slope(series: pd.Series, n: int) -> pd.Series:
        """Slope - 线性回归斜率"""
        def calc_slope(x):
            if len(x) < 2:
                return np.nan
            y = np.arange(len(x))
            mask = ~np.isnan(x)
            if mask.sum() < 2:
                return np.nan
            return np.polyfit(y[mask], x[mask], 1)[0]
        return series.rolling(n, min_periods=2).apply(calc_slope, raw=True)

    @staticmethod
    def rsquare(series: pd.Series, n: int) -> pd.Series:
        """Rsquare - 线性回归 R²"""
        def calc_rsquare(x):
            if len(x) < 2:
                return np.nan
            y = np.arange(len(x))
            mask = ~np.isnan(x)
            if mask.sum() < 2:
                return np.nan
            corr = np.corrcoef(y[mask], x[mask])[0, 1]
            return corr ** 2 if not np.isnan(corr) else np.nan
        return series.rolling(n, min_periods=2).apply(calc_rsquare, raw=True)

    @staticmethod
    def resi(series: pd.Series, n: int) -> pd.Series:
        """Resi - 线性回归残差"""
        def calc_resi(x):
            if len(x) < 2:
                return np.nan
            y = np.arange(len(x))
            mask = ~np.isnan(x)
            if mask.sum() < 2:
                return np.nan
            slope, intercept = np.polyfit(y[mask], x[mask], 1)
            predicted = slope * (len(x) - 1) + intercept
            return x[-1] - predicted
        return series.rolling(n, min_periods=2).apply(calc_resi, raw=True)

    @staticmethod
    def idxmax(series: pd.Series, n: int) -> pd.Series:
        """IdxMax - 最大值位置 (从窗口开始计算)"""
        def calc_idxmax(x):
            if len(x) < 1:
                return np.nan
            return np.nanargmax(x)
        return series.rolling(n, min_periods=1).apply(calc_idxmax, raw=True)

    @staticmethod
    def idxmin(series: pd.Series, n: int) -> pd.Series:
        """IdxMin - 最小值位置 (从窗口开始计算)"""
        def calc_idxmin(x):
            if len(x) < 1:
                return np.nan
            return np.nanargmin(x)
        return series.rolling(n, min_periods=1).apply(calc_idxmin, raw=True)

    @staticmethod
    def skew(series: pd.Series, n: int) -> pd.Series:
        """Skew - 滚动偏度"""
        return series.rolling(n, min_periods=3).skew()

    @staticmethod
    def kurt(series: pd.Series, n: int) -> pd.Series:
        """Kurt - 滚动峰度"""
        return series.rolling(n, min_periods=4).kurt()

    @staticmethod
    def mad(series: pd.Series, n: int) -> pd.Series:
        """Mad - 滚动平均绝对偏差"""
        def calc_mad(x):
            return np.nanmean(np.abs(x - np.nanmean(x)))
        return series.rolling(n, min_periods=1).apply(calc_mad, raw=True)

    @staticmethod
    def prod(series: pd.Series, n: int) -> pd.Series:
        """Prod - 滚动乘积"""
        return series.rolling(n, min_periods=1).apply(np.prod, raw=True)

    # ==================== 指数移动平均 ====================

    @staticmethod
    def ema(series: pd.Series, n: int) -> pd.Series:
        """EMA - 指数移动平均"""
        return series.ewm(span=n, adjust=False).mean()

    @staticmethod
    def wma(series: pd.Series, n: int) -> pd.Series:
        """WMA - 加权移动平均"""
        weights = np.arange(1, n + 1)
        return series.rolling(n).apply(
            lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum(),
            raw=True
        )

    @staticmethod
    def dema(series: pd.Series, n: int) -> pd.Series:
        """DEMA - 双指数移动平均"""
        ema1 = series.ewm(span=n, adjust=False).mean()
        ema2 = ema1.ewm(span=n, adjust=False).mean()
        return 2 * ema1 - ema2

    @staticmethod
    def tema(series: pd.Series, n: int) -> pd.Series:
        """TEMA - 三重指数移动平均"""
        ema1 = series.ewm(span=n, adjust=False).mean()
        ema2 = ema1.ewm(span=n, adjust=False).mean()
        ema3 = ema2.ewm(span=n, adjust=False).mean()
        return 3 * ema1 - 3 * ema2 + ema3

    # ==================== 成对运算符 ====================

    @staticmethod
    def corr(left: pd.Series, right: pd.Series, n: int) -> pd.Series:
        """Corr - 滚动相关系数"""
        return left.rolling(n, min_periods=2).corr(right)

    @staticmethod
    def cov(left: pd.Series, right: pd.Series, n: int) -> pd.Series:
        """Cov - 滚动协方差"""
        return left.rolling(n, min_periods=2).cov(right)

    @staticmethod
    def beta(y: pd.Series, x: pd.Series, n: int) -> pd.Series:
        """Beta - 滚动回归系数"""
        cov_xy = y.rolling(n, min_periods=2).cov(x)
        var_x = x.rolling(n, min_periods=2).var()
        return cov_xy / (var_x + Operators.EPS)

    # ==================== 技术指标 ====================

    @staticmethod
    def rsi(series: pd.Series, n: int = 14) -> pd.Series:
        """RSI - 相对强弱指标"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(n, min_periods=1).mean()
        avg_loss = loss.rolling(n, min_periods=1).mean()
        rs = avg_gain / (avg_loss + Operators.EPS)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD - 移动平均收敛散度"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger(series: pd.Series, n: int = 20, k: float = 2.0):
        """Bollinger Bands - 布林带"""
        middle = series.rolling(n).mean()
        std = series.rolling(n).std()
        upper = middle + k * std
        lower = middle - k * std
        return upper, middle, lower

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
        """ATR - 平均真实波幅"""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        tr = Operators.greater(tr1, Operators.greater(tr2, tr3))
        return tr.rolling(n, min_periods=1).mean()

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
        """ADX - 平均方向指数"""
        atr_val = Operators.atr(high, low, close, n)

        # +DM and -DM
        high_diff = high.diff()
        low_diff = low.shift(1) - low

        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)

        # Smoothed +DI and -DI
        plus_di = 100 * plus_dm.rolling(n).mean() / (atr_val + Operators.EPS)
        minus_di = 100 * minus_dm.rolling(n).mean() / (atr_val + Operators.EPS)

        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + Operators.EPS)
        adx = dx.rolling(n).mean()

        return adx

    # ==================== 重采样运算符 ====================

    @staticmethod
    def resample(series: pd.Series, rule: str, func: str = 'last') -> pd.Series:
        """
        TResample - 时间重采样

        Args:
            series: 输入序列
            rule: 重采样规则 ('1H', '1D', etc.)
            func: 聚合函数 ('last', 'first', 'mean', 'sum', 'max', 'min')
        """
        resampled = series.resample(rule)

        if func == 'last':
            return resampled.last()
        elif func == 'first':
            return resampled.first()
        elif func == 'mean':
            return resampled.mean()
        elif func == 'sum':
            return resampled.sum()
        elif func == 'max':
            return resampled.max()
        elif func == 'min':
            return resampled.min()
        elif func == 'ohlc':
            return resampled.ohlc()
        else:
            raise ValueError(f"Unknown aggregation function: {func}")


# ============================================================
# 表达式解析器
# ============================================================

class ExpressionParser:
    """
    表达式解析器

    解析字符串表达式并计算
    """

    def __init__(self):
        self.ops = Operators()
        self.functions = {
            'abs': self.ops.abs,
            'sign': self.ops.sign,
            'log': self.ops.log,
            'exp': self.ops.exp,
            'sqrt': self.ops.sqrt,
            'mean': self.ops.mean,
            'sum': self.ops.sum,
            'std': self.ops.std,
            'var': self.ops.var,
            'max': self.ops.max,
            'min': self.ops.min,
            'rank': self.ops.rank,
            'delta': self.ops.delta,
            'ref': self.ops.ref,
            'corr': self.ops.corr,
            'cov': self.ops.cov,
            'ema': self.ops.ema,
            'wma': self.ops.wma,
            'rsi': self.ops.rsi,
            'atr': self.ops.atr,
        }

    def parse(self, expr: str, data: pd.DataFrame) -> pd.Series:
        """
        解析并计算表达式

        Args:
            expr: 表达式字符串 (如 "Mean($close, 20)")
            data: 数据 DataFrame

        Returns:
            计算结果
        """
        # 简单的表达式解析
        # TODO: 实现完整的解析器
        logger.warning("Expression parser is in development. Use Operators directly.")
        return pd.Series(dtype=float)


# ============================================================
# 便捷函数
# ============================================================

def calc_expression(
    expr: str,
    data: pd.DataFrame,
) -> pd.Series:
    """
    计算表达式 (便捷函数)

    Args:
        expr: 表达式字符串
        data: 数据

    Returns:
        计算结果
    """
    parser = ExpressionParser()
    return parser.parse(expr, data)


# 导出
__all__ = [
    'Expression',
    'Feature',
    'Constant',
    'UnaryOp',
    'BinaryOp',
    'Operators',
    'ExpressionParser',
    'calc_expression',
]
