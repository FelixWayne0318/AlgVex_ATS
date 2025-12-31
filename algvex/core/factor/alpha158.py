"""
Alpha158 因子库 - 加密货币适配版

基于 Qlib Alpha158 完整实现，适配加密货币数据特点：
- 24/7 交易 (无交易日历限制)
- 高波动性 (需要更鲁棒的计算)
- 永续合约特有数据 (资金费率、持仓量)

因子分类:
- KBAR: K线形态因子 (9个)
- PRICE: 价格因子 (可配置)
- VOLUME: 成交量因子 (可配置)
- ROLLING: 滚动窗口因子 (140个, 28种 × 5窗口)
- CRYPTO: 加密货币特有因子 (11个)

总计: 158+ 因子

用法:
    from algvex.core.factor.alpha158 import Alpha158Calculator

    calc = Alpha158Calculator(windows=[5, 10, 20, 30, 60])
    factors = calc.compute_all(klines_df)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ============================================================
# 表达式运算符 (Qlib ops.py 适配)
# ============================================================

class Operators:
    """
    Qlib 表达式运算符的 Pandas 实现

    适配加密货币高频数据特点
    """

    EPS = 1e-12  # 防止除零

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
    def corr(left: pd.Series, right: pd.Series, n: int) -> pd.Series:
        """Corr - 滚动相关系数"""
        return left.rolling(n, min_periods=2).corr(right)

    @staticmethod
    def cov(left: pd.Series, right: pd.Series, n: int) -> pd.Series:
        """Cov - 滚动协方差"""
        return left.rolling(n, min_periods=2).cov(right)

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
    def ema(series: pd.Series, n: int) -> pd.Series:
        """EMA - 指数移动平均"""
        return series.ewm(span=n, adjust=False).mean()

    @staticmethod
    def wma(series: pd.Series, n: int) -> pd.Series:
        """WMA - 加权移动平均"""
        weights = np.arange(1, n + 1)
        return series.rolling(n).apply(lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum(), raw=True)

    @staticmethod
    def greater(left: pd.Series, right: pd.Series) -> pd.Series:
        """Greater - 逐元素取最大值"""
        return np.maximum(left, right)

    @staticmethod
    def less(left: pd.Series, right: pd.Series) -> pd.Series:
        """Less - 逐元素取最小值"""
        return np.minimum(left, right)


# ============================================================
# Alpha158 因子配置
# ============================================================

@dataclass
class Alpha158Config:
    """Alpha158 因子配置"""

    # 是否启用各类因子
    enable_kbar: bool = True
    enable_price: bool = True
    enable_volume: bool = True
    enable_rolling: bool = True
    enable_crypto: bool = True  # 加密货币特有因子

    # 价格因子配置
    price_windows: List[int] = field(default_factory=lambda: [0])
    price_features: List[str] = field(default_factory=lambda: ['open', 'high', 'low', 'close'])

    # 成交量因子配置
    volume_windows: List[int] = field(default_factory=lambda: [0])

    # 滚动因子配置
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 30, 60])
    rolling_include: List[str] = None  # None = 全部
    rolling_exclude: List[str] = field(default_factory=list)

    # 加密货币因子配置 (周期数，非天数)
    crypto_periods: Dict[str, int] = field(default_factory=lambda: {
        'short': 12,   # 1小时 (5分钟K线)
        'medium': 288,  # 1天
        'long': 288 * 20,  # 20天
    })


# ============================================================
# Alpha158 因子计算器
# ============================================================

class Alpha158Calculator:
    """
    Alpha158 因子计算器 - 加密货币适配版

    完整实现 Qlib Alpha158 的 158 个因子，
    并添加 11 个加密货币特有因子。
    """

    def __init__(self, config: Alpha158Config = None):
        """
        初始化因子计算器

        Args:
            config: Alpha158 配置，None 使用默认配置
        """
        self.config = config or Alpha158Config()
        self.ops = Operators()
        self.EPS = Operators.EPS

    def compute_all(
        self,
        df: pd.DataFrame,
        funding_df: pd.DataFrame = None,
        oi_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        计算所有 Alpha158 因子

        Args:
            df: K线数据 (必须包含 open, high, low, close, volume)
            funding_df: 资金费率数据 (可选)
            oi_df: 持仓量数据 (可选)

        Returns:
            DataFrame: 所有因子
        """
        factors = pd.DataFrame(index=df.index)

        # 确保列名小写
        df = df.copy()
        df.columns = df.columns.str.lower()

        # 1. KBAR 因子 (9个)
        if self.config.enable_kbar:
            kbar_factors = self._compute_kbar(df)
            factors = pd.concat([factors, kbar_factors], axis=1)
            logger.info(f"KBAR factors: {len(kbar_factors.columns)}")

        # 2. PRICE 因子
        if self.config.enable_price:
            price_factors = self._compute_price(df)
            factors = pd.concat([factors, price_factors], axis=1)
            logger.info(f"PRICE factors: {len(price_factors.columns)}")

        # 3. VOLUME 因子
        if self.config.enable_volume:
            volume_factors = self._compute_volume(df)
            factors = pd.concat([factors, volume_factors], axis=1)
            logger.info(f"VOLUME factors: {len(volume_factors.columns)}")

        # 4. ROLLING 因子 (140个)
        if self.config.enable_rolling:
            rolling_factors = self._compute_rolling(df)
            factors = pd.concat([factors, rolling_factors], axis=1)
            logger.info(f"ROLLING factors: {len(rolling_factors.columns)}")

        # 5. CRYPTO 因子 (加密货币特有)
        if self.config.enable_crypto:
            crypto_factors = self._compute_crypto(df, funding_df, oi_df)
            factors = pd.concat([factors, crypto_factors], axis=1)
            logger.info(f"CRYPTO factors: {len(crypto_factors.columns)}")

        logger.info(f"Total Alpha158 factors: {len(factors.columns)}")
        return factors

    def _compute_kbar(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 KBAR 因子 (K线形态, 9个)

        基于日内价格动态的形态特征
        """
        factors = pd.DataFrame(index=df.index)

        o, h, l, c = df['open'], df['high'], df['low'], df['close']
        hl_range = h - l + self.EPS

        # KMID: 收盘-开盘 / 开盘
        factors['KMID'] = (c - o) / (o + self.EPS)

        # KLEN: 最高-最低 / 开盘 (日内波幅)
        factors['KLEN'] = (h - l) / (o + self.EPS)

        # KMID2: 收盘-开盘 / 日内波幅
        factors['KMID2'] = (c - o) / hl_range

        # KUP: 上影线 / 开盘
        factors['KUP'] = (h - self.ops.greater(o, c)) / (o + self.EPS)

        # KUP2: 上影线 / 日内波幅
        factors['KUP2'] = (h - self.ops.greater(o, c)) / hl_range

        # KLOW: 下影线 / 开盘
        factors['KLOW'] = (self.ops.less(o, c) - l) / (o + self.EPS)

        # KLOW2: 下影线 / 日内波幅
        factors['KLOW2'] = (self.ops.less(o, c) - l) / hl_range

        # KSFT: 收盘偏移 / 开盘
        factors['KSFT'] = (2 * c - h - l) / (o + self.EPS)

        # KSFT2: 收盘偏移 / 日内波幅
        factors['KSFT2'] = (2 * c - h - l) / hl_range

        return factors

    def _compute_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 PRICE 因子

        历史价格相对于当前收盘价的比率
        """
        factors = pd.DataFrame(index=df.index)

        c = df['close']

        for feat in self.config.price_features:
            if feat.lower() not in df.columns:
                continue

            series = df[feat.lower()]

            for w in self.config.price_windows:
                name = f"{feat.upper()}{w}"
                if w == 0:
                    factors[name] = series / (c + self.EPS)
                else:
                    factors[name] = self.ops.ref(series, w) / (c + self.EPS)

        return factors

    def _compute_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 VOLUME 因子

        历史成交量相对于当前成交量的比率
        """
        factors = pd.DataFrame(index=df.index)

        v = df['volume']

        for w in self.config.volume_windows:
            name = f"VOLUME{w}"
            if w == 0:
                factors[name] = 1.0  # volume / volume
            else:
                factors[name] = self.ops.ref(v, w) / (v + self.EPS)

        return factors

    def _compute_rolling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 ROLLING 因子 (28种 × 5窗口 = 140个)

        基于滚动窗口的技术指标
        """
        factors = pd.DataFrame(index=df.index)

        c = df['close']
        h = df['high']
        l = df['low']
        v = df['volume']

        windows = self.config.rolling_windows
        include = self.config.rolling_include
        exclude = self.config.rolling_exclude

        def use(x):
            return x not in exclude and (include is None or x in include)

        # ROC - 收益率
        if use('ROC'):
            for w in windows:
                factors[f'ROC{w}'] = self.ops.ref(c, w) / (c + self.EPS)

        # MA - 简单移动平均
        if use('MA'):
            for w in windows:
                factors[f'MA{w}'] = self.ops.mean(c, w) / (c + self.EPS)

        # STD - 标准差
        if use('STD'):
            for w in windows:
                factors[f'STD{w}'] = self.ops.std(c, w) / (c + self.EPS)

        # BETA - 线性回归斜率
        if use('BETA'):
            for w in windows:
                factors[f'BETA{w}'] = self.ops.slope(c, w) / (c + self.EPS)

        # RSQR - R²
        if use('RSQR'):
            for w in windows:
                factors[f'RSQR{w}'] = self.ops.rsquare(c, w)

        # RESI - 回归残差
        if use('RESI'):
            for w in windows:
                factors[f'RESI{w}'] = self.ops.resi(c, w) / (c + self.EPS)

        # MAX - 最高价
        if use('MAX'):
            for w in windows:
                factors[f'MAX{w}'] = self.ops.max(h, w) / (c + self.EPS)

        # MIN - 最低价
        if use('MIN'):
            for w in windows:
                factors[f'MIN{w}'] = self.ops.min(l, w) / (c + self.EPS)

        # QTLU - 80%分位数
        if use('QTLU'):
            for w in windows:
                factors[f'QTLU{w}'] = self.ops.quantile(c, w, 0.8) / (c + self.EPS)

        # QTLD - 20%分位数
        if use('QTLD'):
            for w in windows:
                factors[f'QTLD{w}'] = self.ops.quantile(c, w, 0.2) / (c + self.EPS)

        # RANK - 百分位排名
        if use('RANK'):
            for w in windows:
                factors[f'RANK{w}'] = self.ops.rank(c, w)

        # RSV - 相对强弱值 (类似KDJ)
        if use('RSV'):
            for w in windows:
                min_low = self.ops.min(l, w)
                max_high = self.ops.max(h, w)
                factors[f'RSV{w}'] = (c - min_low) / (max_high - min_low + self.EPS)

        # IMAX - 最大值位置
        if use('IMAX'):
            for w in windows:
                factors[f'IMAX{w}'] = self.ops.idxmax(h, w) / w

        # IMIN - 最小值位置
        if use('IMIN'):
            for w in windows:
                factors[f'IMIN{w}'] = self.ops.idxmin(l, w) / w

        # IMXD - 最大最小值位置差
        if use('IMXD'):
            for w in windows:
                factors[f'IMXD{w}'] = (self.ops.idxmax(h, w) - self.ops.idxmin(l, w)) / w

        # CORR - 价量相关性
        if use('CORR'):
            log_v = np.log(v + 1)
            for w in windows:
                factors[f'CORR{w}'] = self.ops.corr(c, log_v, w)

        # CORD - 价量变化相关性
        if use('CORD'):
            c_ret = c / self.ops.ref(c, 1)
            v_ret = np.log(v / self.ops.ref(v, 1) + 1)
            for w in windows:
                factors[f'CORD{w}'] = self.ops.corr(c_ret, v_ret, w)

        # CNTP - 上涨天数占比
        if use('CNTP'):
            up = (c > self.ops.ref(c, 1)).astype(float)
            for w in windows:
                factors[f'CNTP{w}'] = self.ops.mean(up, w)

        # CNTN - 下跌天数占比
        if use('CNTN'):
            down = (c < self.ops.ref(c, 1)).astype(float)
            for w in windows:
                factors[f'CNTN{w}'] = self.ops.mean(down, w)

        # CNTD - 涨跌天数差
        if use('CNTD'):
            up = (c > self.ops.ref(c, 1)).astype(float)
            down = (c < self.ops.ref(c, 1)).astype(float)
            for w in windows:
                factors[f'CNTD{w}'] = self.ops.mean(up, w) - self.ops.mean(down, w)

        # SUMP - 上涨幅度占比 (类似RSI)
        if use('SUMP'):
            change = c - self.ops.ref(c, 1)
            gain = self.ops.greater(change, pd.Series(0, index=c.index))
            abs_change = np.abs(change)
            for w in windows:
                factors[f'SUMP{w}'] = self.ops.sum(gain, w) / (self.ops.sum(abs_change, w) + self.EPS)

        # SUMN - 下跌幅度占比
        if use('SUMN'):
            change = c - self.ops.ref(c, 1)
            loss = self.ops.greater(-change, pd.Series(0, index=c.index))
            abs_change = np.abs(change)
            for w in windows:
                factors[f'SUMN{w}'] = self.ops.sum(loss, w) / (self.ops.sum(abs_change, w) + self.EPS)

        # SUMD - 涨跌幅度差
        if use('SUMD'):
            change = c - self.ops.ref(c, 1)
            gain = self.ops.greater(change, pd.Series(0, index=c.index))
            loss = self.ops.greater(-change, pd.Series(0, index=c.index))
            abs_change = np.abs(change)
            for w in windows:
                factors[f'SUMD{w}'] = (self.ops.sum(gain, w) - self.ops.sum(loss, w)) / (self.ops.sum(abs_change, w) + self.EPS)

        # VMA - 成交量移动平均
        if use('VMA'):
            for w in windows:
                factors[f'VMA{w}'] = self.ops.mean(v, w) / (v + self.EPS)

        # VSTD - 成交量标准差
        if use('VSTD'):
            for w in windows:
                factors[f'VSTD{w}'] = self.ops.std(v, w) / (v + self.EPS)

        # WVMA - 加权成交量波动
        if use('WVMA'):
            c_ret = np.abs(c / self.ops.ref(c, 1) - 1)
            weighted = c_ret * v
            for w in windows:
                factors[f'WVMA{w}'] = self.ops.std(weighted, w) / (self.ops.mean(weighted, w) + self.EPS)

        # VSUMP - 成交量上涨占比
        if use('VSUMP'):
            v_change = v - self.ops.ref(v, 1)
            v_gain = self.ops.greater(v_change, pd.Series(0, index=v.index))
            v_abs = np.abs(v_change)
            for w in windows:
                factors[f'VSUMP{w}'] = self.ops.sum(v_gain, w) / (self.ops.sum(v_abs, w) + self.EPS)

        # VSUMN - 成交量下跌占比
        if use('VSUMN'):
            v_change = v - self.ops.ref(v, 1)
            v_loss = self.ops.greater(-v_change, pd.Series(0, index=v.index))
            v_abs = np.abs(v_change)
            for w in windows:
                factors[f'VSUMN{w}'] = self.ops.sum(v_loss, w) / (self.ops.sum(v_abs, w) + self.EPS)

        # VSUMD - 成交量涨跌差
        if use('VSUMD'):
            v_change = v - self.ops.ref(v, 1)
            v_gain = self.ops.greater(v_change, pd.Series(0, index=v.index))
            v_loss = self.ops.greater(-v_change, pd.Series(0, index=v.index))
            v_abs = np.abs(v_change)
            for w in windows:
                factors[f'VSUMD{w}'] = (self.ops.sum(v_gain, w) - self.ops.sum(v_loss, w)) / (self.ops.sum(v_abs, w) + self.EPS)

        return factors

    def _compute_crypto(
        self,
        df: pd.DataFrame,
        funding_df: pd.DataFrame = None,
        oi_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        计算 CRYPTO 因子 (加密货币特有, 11个)

        包括资金费率、持仓量等永续合约特有数据
        """
        factors = pd.DataFrame(index=df.index)

        c = df['close']
        v = df['volume']

        periods = self.config.crypto_periods
        short = periods.get('short', 12)
        medium = periods.get('medium', 288)
        long_p = periods.get('long', 288 * 20)

        # 1. RETURN_SHORT - 短期收益率
        factors['RETURN_SHORT'] = c.pct_change(short)

        # 2. RETURN_MEDIUM - 中期收益率
        factors['RETURN_MEDIUM'] = c.pct_change(medium)

        # 3. MA_CROSS - 均线交叉
        ma_fast = self.ops.mean(c, short)
        ma_slow = self.ops.mean(c, medium)
        factors['MA_CROSS'] = (ma_fast - ma_slow) / (ma_slow + self.EPS)

        # 4. BREAKOUT - 突破信号
        high_long = self.ops.max(df['high'], long_p)
        low_long = self.ops.min(df['low'], long_p)
        factors['BREAKOUT'] = (c - low_long) / (high_long - low_long + self.EPS)

        # 5. TREND_STRENGTH - 趋势强度
        factors['TREND_STRENGTH'] = self.ops.rsquare(c, medium)

        # 6. ATR - 平均真实波幅
        tr = self.ops.greater(
            df['high'] - df['low'],
            self.ops.greater(
                np.abs(df['high'] - self.ops.ref(c, 1)),
                np.abs(df['low'] - self.ops.ref(c, 1))
            )
        )
        factors['ATR'] = self.ops.mean(tr, medium) / (c + self.EPS)

        # 7. REALIZED_VOL - 实现波动率
        returns = np.log(c / self.ops.ref(c, 1))
        factors['REALIZED_VOL'] = self.ops.std(returns, medium) * np.sqrt(medium)

        # 8. VOL_REGIME - 波动率状态
        short_vol = self.ops.std(returns, short)
        long_vol = self.ops.std(returns, medium)
        factors['VOL_REGIME'] = short_vol / (long_vol + self.EPS)

        # 资金费率因子 (如果有数据)
        if funding_df is not None and len(funding_df) > 0:
            if 'funding_rate' in funding_df.columns:
                # 对齐到K线时间
                funding = funding_df['funding_rate'].reindex(df.index, method='ffill')

                # 9. FUNDING_MOMENTUM - 资金费率动量
                factors['FUNDING_MOMENTUM'] = self.ops.mean(funding, short)

                # 10. FUNDING_ZSCORE - 资金费率Z分数
                fr_mean = self.ops.mean(funding, medium)
                fr_std = self.ops.std(funding, medium)
                factors['FUNDING_ZSCORE'] = (funding - fr_mean) / (fr_std + self.EPS)

        # 持仓量因子 (如果有数据)
        if oi_df is not None and len(oi_df) > 0:
            oi_col = 'open_interest' if 'open_interest' in oi_df.columns else 'sumOpenInterest'
            if oi_col in oi_df.columns:
                # 对齐到K线时间
                oi = oi_df[oi_col].reindex(df.index, method='ffill')

                # 11. OI_CHANGE_RATE - 持仓量变化率
                factors['OI_CHANGE_RATE'] = oi.pct_change(short)

                # 如果同时有资金费率，计算背离
                if 'FUNDING_MOMENTUM' in factors.columns:
                    # OI_FUNDING_DIVERGENCE - 持仓与资金费率背离
                    oi_norm = (oi - self.ops.mean(oi, medium)) / (self.ops.std(oi, medium) + self.EPS)
                    fr_norm = factors['FUNDING_ZSCORE'] if 'FUNDING_ZSCORE' in factors.columns else 0
                    factors['OI_FUNDING_DIVERGENCE'] = oi_norm - fr_norm

        return factors

    def get_factor_names(self) -> List[str]:
        """获取所有因子名称"""
        names = []

        if self.config.enable_kbar:
            names.extend(['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2'])

        if self.config.enable_price:
            for feat in self.config.price_features:
                for w in self.config.price_windows:
                    names.append(f"{feat.upper()}{w}")

        if self.config.enable_volume:
            for w in self.config.volume_windows:
                names.append(f"VOLUME{w}")

        if self.config.enable_rolling:
            rolling_types = [
                'ROC', 'MA', 'STD', 'BETA', 'RSQR', 'RESI', 'MAX', 'MIN',
                'QTLU', 'QTLD', 'RANK', 'RSV', 'IMAX', 'IMIN', 'IMXD',
                'CORR', 'CORD', 'CNTP', 'CNTN', 'CNTD', 'SUMP', 'SUMN', 'SUMD',
                'VMA', 'VSTD', 'WVMA', 'VSUMP', 'VSUMN', 'VSUMD'
            ]
            for t in rolling_types:
                if t not in self.config.rolling_exclude:
                    for w in self.config.rolling_windows:
                        names.append(f"{t}{w}")

        if self.config.enable_crypto:
            names.extend([
                'RETURN_SHORT', 'RETURN_MEDIUM', 'MA_CROSS', 'BREAKOUT',
                'TREND_STRENGTH', 'ATR', 'REALIZED_VOL', 'VOL_REGIME',
                'FUNDING_MOMENTUM', 'FUNDING_ZSCORE', 'OI_CHANGE_RATE', 'OI_FUNDING_DIVERGENCE'
            ])

        return names


# ============================================================
# 便捷函数
# ============================================================

def get_alpha158_calculator(
    windows: List[int] = None,
    include_crypto: bool = True,
) -> Alpha158Calculator:
    """
    获取 Alpha158 因子计算器

    Args:
        windows: 滚动窗口列表，None 使用默认 [5, 10, 20, 30, 60]
        include_crypto: 是否包含加密货币特有因子

    Returns:
        Alpha158Calculator 实例
    """
    config = Alpha158Config(
        rolling_windows=windows or [5, 10, 20, 30, 60],
        enable_crypto=include_crypto,
    )
    return Alpha158Calculator(config)


def compute_alpha158(
    klines: pd.DataFrame,
    funding: pd.DataFrame = None,
    oi: pd.DataFrame = None,
    windows: List[int] = None,
) -> pd.DataFrame:
    """
    计算 Alpha158 因子 (便捷函数)

    Args:
        klines: K线数据
        funding: 资金费率数据 (可选)
        oi: 持仓量数据 (可选)
        windows: 滚动窗口列表

    Returns:
        DataFrame: 所有因子
    """
    calc = get_alpha158_calculator(windows)
    return calc.compute_all(klines, funding, oi)
