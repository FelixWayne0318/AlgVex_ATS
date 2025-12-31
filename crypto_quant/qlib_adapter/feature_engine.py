"""
加密货币因子引擎 - 永续合约专用因子库

因子分类:
1. 价格动量因子
2. 波动率因子
3. 成交量因子
4. 资金费率因子 (永续特有)
5. 持仓量因子 (永续特有)
6. 多空博弈因子 (永续特有)
7. 订单簿微观因子
8. 情绪因子
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from loguru import logger


class CryptoFeatureEngine:
    """
    加密货币因子计算引擎

    使用方法:
    1. 传入Qlib格式数据
    2. 调用calculate_all_features()计算所有因子
    3. 或调用特定因子组
    """

    def __init__(self):
        # 因子定义: (表达式/函数, 因子名, 因子描述)
        self.factor_definitions = self._define_factors()

    def _define_factors(self) -> Dict[str, List[Tuple]]:
        """定义所有因子"""
        return {
            "momentum": self._momentum_factors(),
            "volatility": self._volatility_factors(),
            "volume": self._volume_factors(),
            "funding": self._funding_rate_factors(),
            "oi": self._open_interest_factors(),
            "ls_ratio": self._long_short_factors(),
            "taker": self._taker_factors(),
            "sentiment": self._sentiment_factors(),
            "composite": self._composite_factors(),
        }

    # ==================== 动量因子 ====================
    def _momentum_factors(self) -> List[Tuple]:
        return [
            ("return_1h", "$close / $close.shift(1) - 1", "1小时收益"),
            ("return_4h", "$close / $close.shift(4) - 1", "4小时收益"),
            ("return_24h", "$close / $close.shift(24) - 1", "24小时收益"),
            ("return_7d", "$close / $close.shift(168) - 1", "7日收益"),

            ("mom_12h", "$close / $close.shift(12) - 1", "12小时动量"),
            ("mom_24h", "$close / $close.shift(24) - 1", "24小时动量"),

            ("roc_12h", "($close - $close.shift(12)) / $close.shift(12)", "12小时变化率"),

            ("ma_cross_12_24", "$close.rolling(12).mean() / $close.rolling(24).mean() - 1", "均线交叉12/24"),
            ("ma_cross_24_72", "$close.rolling(24).mean() / $close.rolling(72).mean() - 1", "均线交叉24/72"),

            ("price_position", "($close - $low.rolling(24).min()) / ($high.rolling(24).max() - $low.rolling(24).min())", "价格位置"),
        ]

    # ==================== 波动率因子 ====================
    def _volatility_factors(self) -> List[Tuple]:
        return [
            ("volatility_12h", "$return_1h.rolling(12).std()", "12小时波动率"),
            ("volatility_24h", "$return_1h.rolling(24).std()", "24小时波动率"),
            ("volatility_7d", "$return_1h.rolling(168).std()", "7日波动率"),

            ("volatility_ratio", "$return_1h.rolling(12).std() / $return_1h.rolling(72).std()", "波动率比率"),

            ("atr_24h", "(($high - $low).rolling(24).mean()) / $close", "24小时ATR比率"),

            ("range_24h", "($high.rolling(24).max() - $low.rolling(24).min()) / $close", "24小时振幅"),

            ("skewness_24h", "$return_1h.rolling(24).skew()", "24小时偏度"),
            ("kurtosis_24h", "$return_1h.rolling(24).kurt()", "24小时峰度"),
        ]

    # ==================== 成交量因子 ====================
    def _volume_factors(self) -> List[Tuple]:
        return [
            ("volume_ratio_12h", "$volume / $volume.rolling(12).mean()", "12小时量比"),
            ("volume_ratio_24h", "$volume / $volume.rolling(24).mean()", "24小时量比"),

            ("volume_ma_12", "$volume.rolling(12).mean()", "12小时成交量均值"),

            ("volume_std_24h", "$volume.rolling(24).std() / $volume.rolling(24).mean()", "成交量波动"),

            ("volume_trend", "$volume.rolling(12).mean() / $volume.rolling(72).mean() - 1", "成交量趋势"),

            ("price_volume_corr", "$close.rolling(24).corr($volume)", "价量相关性"),

            ("obv_change", "(($close > $close.shift(1)).astype(int) * 2 - 1) * $volume", "OBV变化"),
        ]

    # ==================== 资金费率因子 (永续特有) ====================
    def _funding_rate_factors(self) -> List[Tuple]:
        return [
            ("funding_rate", "$funding_rate", "资金费率"),

            ("funding_rate_ma_7", "$funding_rate.rolling(21).mean()", "7日资金费率均值(3次/天)"),
            ("funding_rate_ma_30", "$funding_rate.rolling(90).mean()", "30日资金费率均值"),

            ("funding_rate_std", "$funding_rate.rolling(21).std()", "资金费率波动"),

            ("funding_rate_zscore", "($funding_rate - $funding_rate.rolling(90).mean()) / $funding_rate.rolling(90).std()", "资金费率Z分数"),

            ("funding_rate_extreme_high", "($funding_rate > 0.001).astype(int)", "极高资金费率(>0.1%)"),
            ("funding_rate_extreme_low", "($funding_rate < -0.001).astype(int)", "极低资金费率(<-0.1%)"),

            ("funding_rate_cumsum_7d", "$funding_rate.rolling(21).sum()", "7日累计资金费率"),

            ("funding_rate_positive_ratio", "($funding_rate > 0).rolling(21).mean()", "正费率占比"),

            # 资金费率与价格关系
            ("funding_price_corr", "$funding_rate.rolling(72).corr($close.pct_change())", "费率价格相关性"),
        ]

    # ==================== 持仓量因子 (永续特有) ====================
    def _open_interest_factors(self) -> List[Tuple]:
        return [
            ("oi", "$open_interest", "持仓量"),

            ("oi_change_1h", "$open_interest / $open_interest.shift(1) - 1", "1小时持仓变化"),
            ("oi_change_24h", "$open_interest / $open_interest.shift(24) - 1", "24小时持仓变化"),

            ("oi_ma_ratio", "$open_interest / $open_interest.rolling(72).mean()", "持仓量均值比"),

            ("oi_percentile", "($open_interest - $open_interest.rolling(168).min()) / ($open_interest.rolling(168).max() - $open_interest.rolling(168).min())", "持仓量分位"),

            # 价仓关系
            ("oi_price_corr", "$open_interest.rolling(24).corr($close)", "价仓相关性"),

            # 价涨仓增 = 趋势确认
            ("oi_trend_confirm", "(($close > $close.shift(1)) & ($open_interest > $open_interest.shift(1))).astype(int)", "价仓趋势确认"),

            # 价涨仓减 = 空头回补
            ("oi_short_cover", "(($close > $close.shift(1)) & ($open_interest < $open_interest.shift(1))).astype(int)", "空头回补信号"),

            # 持仓/成交比
            ("oi_volume_ratio", "$open_interest_value / ($quote_volume + 1)", "持仓成交比"),
        ]

    # ==================== 多空博弈因子 (永续特有) ====================
    def _long_short_factors(self) -> List[Tuple]:
        return [
            ("ls_ratio", "$ls_ratio", "多空比"),

            ("ls_ratio_change", "$ls_ratio - $ls_ratio.shift(1)", "多空比变化"),

            ("ls_ratio_ma", "$ls_ratio.rolling(24).mean()", "多空比均值"),

            ("ls_ratio_zscore", "($ls_ratio - $ls_ratio.rolling(72).mean()) / $ls_ratio.rolling(72).std()", "多空比Z分数"),

            # 极端多空
            ("ls_extreme_long", "($ls_ratio > $ls_ratio.rolling(168).quantile(0.9)).astype(int)", "极端多头"),
            ("ls_extreme_short", "($ls_ratio < $ls_ratio.rolling(168).quantile(0.1)).astype(int)", "极端空头"),

            # 大户 vs 散户背离 (逆向指标)
            ("ls_divergence", "$ls_ratio - $top_ls_ratio", "散户大户背离"),

            # 多空与价格
            ("ls_price_corr", "$ls_ratio.rolling(24).corr($close.pct_change())", "多空价格相关"),
        ]

    # ==================== 主动买卖因子 ====================
    def _taker_factors(self) -> List[Tuple]:
        return [
            ("taker_ratio", "$taker_buy_sell_ratio", "主动买卖比"),

            ("taker_imbalance", "($taker_buy_vol - $taker_sell_vol) / ($taker_buy_vol + $taker_sell_vol)", "买卖不平衡"),

            ("taker_ratio_ma", "$taker_buy_sell_ratio.rolling(24).mean()", "主动买卖比均值"),

            ("taker_momentum", "$taker_buy_sell_ratio.rolling(6).mean() - $taker_buy_sell_ratio.rolling(24).mean()", "主动买卖动量"),
        ]

    # ==================== 情绪因子 ====================
    def _sentiment_factors(self) -> List[Tuple]:
        return [
            ("fear_greed", "$fear_greed_index", "恐惧贪婪指数"),

            ("fear_greed_ma_7d", "$fear_greed_index.rolling(168).mean()", "7日情绪均值"),

            ("fear_greed_extreme_fear", "($fear_greed_index < 25).astype(int)", "极度恐惧"),
            ("fear_greed_extreme_greed", "($fear_greed_index > 75).astype(int)", "极度贪婪"),

            ("fear_greed_change", "$fear_greed_index - $fear_greed_index.shift(24)", "情绪变化"),
        ]

    # ==================== 复合因子 ====================
    def _composite_factors(self) -> List[Tuple]:
        return [
            # 趋势强度 = 动量 + 持仓确认
            ("trend_strength", "mom_24h * oi_trend_confirm", "趋势强度"),

            # 资金压力 = 高费率 + 高持仓
            ("funding_pressure", "funding_rate_zscore * oi_ma_ratio", "资金压力"),

            # 情绪极值 = 多空极端 + 恐惧贪婪极端
            ("sentiment_extreme", "(ls_extreme_long + fear_greed_extreme_greed) / 2", "情绪极值"),

            # 反转信号 = 资金费率极高 + 多头拥挤
            ("reversal_signal", "funding_rate_extreme_high * ls_extreme_long", "反转信号"),
        ]

    # ==================== 计算引擎 ====================
    def calculate_factor(self, df: pd.DataFrame, expression: str, name: str) -> pd.Series:
        """计算单个因子"""
        try:
            # 替换$为df列引用
            expr = expression
            for col in df.columns:
                if col.startswith("$"):
                    expr = expr.replace(col, f"df['{col}']")

            # 执行表达式
            result = eval(expr)

            if isinstance(result, pd.Series):
                return result
            elif isinstance(result, pd.DataFrame):
                return result.iloc[:, 0]
            else:
                return pd.Series(result, index=df.index)

        except Exception as e:
            logger.warning(f"Failed to calculate factor {name}: {e}")
            return pd.Series(np.nan, index=df.index)

    def calculate_group_factors(
        self,
        df: pd.DataFrame,
        group: str,
    ) -> pd.DataFrame:
        """计算一组因子"""
        if group not in self.factor_definitions:
            raise ValueError(f"Unknown factor group: {group}")

        factors = self.factor_definitions[group]
        result = df.copy()

        for name, expression, description in factors:
            result[name] = self.calculate_factor(df, expression, name)

        return result

    def calculate_all_features(
        self,
        df: pd.DataFrame,
        groups: List[str] = None,
    ) -> pd.DataFrame:
        """
        计算所有因子

        Args:
            df: 输入数据 (Qlib格式)
            groups: 要计算的因子组，None表示全部

        Returns:
            包含所有因子的DataFrame
        """
        if groups is None:
            groups = list(self.factor_definitions.keys())

        result = df.copy()

        # 检查是否是MultiIndex
        is_multi_index = isinstance(result.index, pd.MultiIndex) and "instrument" in result.index.names

        if is_multi_index:
            # 按instrument分组计算
            instruments = result.index.get_level_values("instrument").unique()
        else:
            # 单标的模式
            instruments = ["default"]
            result["instrument"] = "default"

        all_results = []
        for instrument in instruments:
            if is_multi_index:
                inst_df = result.xs(instrument, level="instrument").copy()
            else:
                inst_df = result.drop(columns=["instrument"]).copy()

            for group in groups:
                if group == "composite":
                    continue  # 复合因子最后计算

                factors = self.factor_definitions[group]
                for name, expression, description in factors:
                    try:
                        # 简化表达式计算
                        inst_df[name] = self._eval_expression(inst_df, expression)
                    except Exception as e:
                        logger.debug(f"Factor {name} failed for {instrument}: {e}")
                        inst_df[name] = np.nan

            inst_df["instrument"] = instrument
            # 确保index有名字
            if inst_df.index.name is None:
                inst_df.index.name = "datetime"
            all_results.append(inst_df.reset_index())

        result = pd.concat(all_results, ignore_index=True)
        if "datetime" in result.columns and "instrument" in result.columns:
            result = result.set_index(["datetime", "instrument"])

        logger.info(f"Calculated {len(result.columns)} features")
        return result

    def _eval_expression(self, df: pd.DataFrame, expr: str) -> pd.Series:
        """安全计算表达式"""
        # 创建本地变量空间
        local_vars = {}
        for col in df.columns:
            if col.startswith("$"):
                var_name = col.replace("$", "")
                local_vars[var_name] = df[col]
            else:
                local_vars[col] = df[col]

        # 替换表达式中的$
        safe_expr = expr.replace("$", "")

        try:
            return eval(safe_expr, {"__builtins__": {}}, {**local_vars, "pd": pd, "np": np})
        except:
            return pd.Series(np.nan, index=df.index)

    def get_feature_list(self, groups: List[str] = None) -> List[str]:
        """获取因子列表"""
        if groups is None:
            groups = list(self.factor_definitions.keys())

        features = []
        for group in groups:
            for name, _, _ in self.factor_definitions.get(group, []):
                features.append(name)

        return features


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 创建示例数据
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    instruments = ["btcusdt", "ethusdt"]

    data = []
    for inst in instruments:
        for dt in dates:
            data.append({
                "datetime": dt,
                "instrument": inst,
                "$open": 40000 + np.random.randn() * 1000,
                "$high": 40500 + np.random.randn() * 1000,
                "$low": 39500 + np.random.randn() * 1000,
                "$close": 40000 + np.random.randn() * 1000,
                "$volume": 1000000 + np.random.randn() * 100000,
                "$funding_rate": 0.0001 + np.random.randn() * 0.0005,
                "$open_interest": 500000 + np.random.randn() * 50000,
                "$ls_ratio": 1.0 + np.random.randn() * 0.2,
            })

    df = pd.DataFrame(data).set_index(["datetime", "instrument"])

    # 计算因子
    engine = CryptoFeatureEngine()
    result = engine.calculate_all_features(df, groups=["momentum", "funding", "oi"])

    print("计算完成的因子:")
    print(result.columns.tolist())
    print(result.head())
