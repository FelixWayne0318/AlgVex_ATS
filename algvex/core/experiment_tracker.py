"""
AlgVex 实验追踪器 (MLflow 集成)

功能:
- 自动记录实验参数、指标、模型
- 支持实验对比和可视化
- 与 Qlib 风格保持一致

用法:
    from algvex.core.experiment_tracker import ExperimentTracker

    # 初始化
    tracker = ExperimentTracker(
        experiment_name="crypto_backtest",
        tracking_uri="./mlruns"
    )

    # 开始实验
    with tracker.start_run(run_name="btc_momentum_v1"):
        # 记录参数
        tracker.log_params({
            "symbol": "BTCUSDT",
            "leverage": 3.0,
            "model": "LightGBM"
        })

        # ... 执行回测 ...

        # 记录指标
        tracker.log_metrics({
            "sharpe_ratio": 1.85,
            "max_drawdown": 0.12,
            "total_return": 0.45
        })

        # 记录模型
        tracker.log_model(model, "lightgbm_model")

        # 记录图表
        tracker.log_figure(fig, "equity_curve.png")
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from contextlib import contextmanager

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """实验结果数据类"""
    run_id: str
    run_name: str
    experiment_name: str
    start_time: datetime
    end_time: Optional[datetime]
    params: Dict[str, Any]
    metrics: Dict[str, float]
    artifacts: List[str]
    status: str  # "RUNNING", "FINISHED", "FAILED"

    def to_dict(self) -> Dict:
        result = asdict(self)
        result['start_time'] = self.start_time.isoformat()
        result['end_time'] = self.end_time.isoformat() if self.end_time else None
        return result


class ExperimentTracker:
    """
    实验追踪器 (Qlib/MLflow 风格)

    特性:
    - 自动参数记录
    - 指标追踪
    - 模型版本管理
    - 实验对比
    """

    def __init__(
        self,
        experiment_name: str = "algvex_experiment",
        tracking_uri: str = "./mlruns",
        auto_log: bool = True,
    ):
        """
        初始化追踪器

        Args:
            experiment_name: 实验名称
            tracking_uri: MLflow 追踪 URI
            auto_log: 是否自动记录
        """
        self.experiment_name = experiment_name
        self.tracking_uri = Path(tracking_uri).expanduser().absolute()
        self.auto_log = auto_log

        self._current_run = None
        self._run_params = {}
        self._run_metrics = {}
        self._run_artifacts = []

        if MLFLOW_AVAILABLE:
            # 设置 MLflow
            mlflow.set_tracking_uri(str(self.tracking_uri))
            mlflow.set_experiment(experiment_name)
            self._client = MlflowClient(str(self.tracking_uri))
            logger.info(f"MLflow initialized: {self.tracking_uri}")
        else:
            logger.warning("MLflow not installed, using local file tracking")
            self._client = None
            # 创建本地追踪目录
            self.tracking_uri.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def start_run(
        self,
        run_name: str = None,
        tags: Dict[str, str] = None,
        nested: bool = False,
    ):
        """
        开始一个实验运行

        Args:
            run_name: 运行名称
            tags: 标签
            nested: 是否嵌套运行

        Yields:
            运行上下文
        """
        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self._run_params = {}
        self._run_metrics = {}
        self._run_artifacts = []
        start_time = datetime.now()

        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=run_name, nested=nested) as run:
                self._current_run = run
                if tags:
                    mlflow.set_tags(tags)
                try:
                    yield run
                    mlflow.set_tag("status", "FINISHED")
                except Exception as e:
                    mlflow.set_tag("status", "FAILED")
                    mlflow.set_tag("error", str(e))
                    raise
                finally:
                    self._current_run = None
        else:
            # 本地追踪模式
            run_id = f"local_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            try:
                yield {"run_id": run_id, "run_name": run_name}
                status = "FINISHED"
            except Exception as e:
                status = "FAILED"
                raise
            finally:
                # 保存到本地文件
                result = ExperimentResult(
                    run_id=run_id,
                    run_name=run_name,
                    experiment_name=self.experiment_name,
                    start_time=start_time,
                    end_time=datetime.now(),
                    params=self._run_params,
                    metrics=self._run_metrics,
                    artifacts=self._run_artifacts,
                    status=status,
                )
                self._save_local_run(result)

    def log_params(self, params: Dict[str, Any]):
        """
        记录参数

        Args:
            params: 参数字典
        """
        self._run_params.update(params)

        if MLFLOW_AVAILABLE and self._current_run:
            # MLflow 只接受字符串值
            str_params = {k: str(v) for k, v in params.items()}
            mlflow.log_params(str_params)
        else:
            logger.debug(f"Logged params: {params}")

    def log_param(self, key: str, value: Any):
        """记录单个参数"""
        self.log_params({key: value})

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        记录指标

        Args:
            metrics: 指标字典
            step: 步骤编号
        """
        self._run_metrics.update(metrics)

        if MLFLOW_AVAILABLE and self._current_run:
            mlflow.log_metrics(metrics, step=step)
        else:
            logger.debug(f"Logged metrics: {metrics}")

    def log_metric(self, key: str, value: float, step: int = None):
        """记录单个指标"""
        self.log_metrics({key: value}, step=step)

    def log_artifact(self, local_path: str, artifact_path: str = None):
        """
        记录文件

        Args:
            local_path: 本地文件路径
            artifact_path: 存储路径
        """
        self._run_artifacts.append(local_path)

        if MLFLOW_AVAILABLE and self._current_run:
            mlflow.log_artifact(local_path, artifact_path)
        else:
            logger.debug(f"Logged artifact: {local_path}")

    def log_figure(self, fig, filename: str):
        """
        记录图表

        Args:
            fig: matplotlib 或 plotly 图表
            filename: 文件名
        """
        # 临时保存图表
        temp_path = Path("/tmp") / filename
        temp_path.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(fig, 'savefig'):
            # matplotlib
            fig.savefig(temp_path, dpi=150, bbox_inches='tight')
        elif hasattr(fig, 'write_image'):
            # plotly
            fig.write_image(str(temp_path))
        elif hasattr(fig, 'write_html'):
            # plotly (HTML)
            temp_path = temp_path.with_suffix('.html')
            fig.write_html(str(temp_path))
        else:
            logger.warning(f"Unknown figure type: {type(fig)}")
            return

        self.log_artifact(str(temp_path))

    def log_model(self, model, artifact_path: str = "model"):
        """
        记录模型

        Args:
            model: 模型对象
            artifact_path: 存储路径
        """
        if MLFLOW_AVAILABLE and self._current_run:
            # 尝试自动检测模型类型
            model_type = type(model).__module__.split('.')[0]

            if model_type == 'lightgbm':
                mlflow.lightgbm.log_model(model, artifact_path)
            elif model_type == 'xgboost':
                mlflow.xgboost.log_model(model, artifact_path)
            elif model_type == 'sklearn':
                mlflow.sklearn.log_model(model, artifact_path)
            elif model_type == 'torch':
                mlflow.pytorch.log_model(model, artifact_path)
            else:
                # 通用方式
                import pickle
                temp_path = Path("/tmp") / f"{artifact_path}.pkl"
                with open(temp_path, 'wb') as f:
                    pickle.dump(model, f)
                self.log_artifact(str(temp_path), artifact_path)
        else:
            logger.debug(f"Logged model to {artifact_path}")

    def log_dataframe(self, df, filename: str):
        """
        记录 DataFrame

        Args:
            df: pandas DataFrame
            filename: 文件名
        """
        temp_path = Path("/tmp") / filename

        if filename.endswith('.parquet'):
            df.to_parquet(temp_path)
        elif filename.endswith('.csv'):
            df.to_csv(temp_path, index=False)
        else:
            df.to_parquet(temp_path.with_suffix('.parquet'))

        self.log_artifact(str(temp_path))

    def log_dict(self, data: Dict, filename: str):
        """
        记录字典为 JSON

        Args:
            data: 字典数据
            filename: 文件名
        """
        temp_path = Path("/tmp") / filename
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        self.log_artifact(str(temp_path))

    def _save_local_run(self, result: ExperimentResult):
        """保存运行结果到本地"""
        run_dir = self.tracking_uri / self.experiment_name / result.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # 保存元数据
        with open(run_dir / "meta.json", 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Run saved to {run_dir}")

    def get_experiment_runs(self, max_results: int = 100) -> List[Dict]:
        """获取实验运行列表"""
        if MLFLOW_AVAILABLE and self._client:
            experiment = self._client.get_experiment_by_name(self.experiment_name)
            if experiment:
                runs = self._client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    max_results=max_results,
                    order_by=["start_time DESC"]
                )
                return [
                    {
                        "run_id": r.info.run_id,
                        "run_name": r.info.run_name,
                        "status": r.info.status,
                        "start_time": r.info.start_time,
                        "metrics": r.data.metrics,
                        "params": r.data.params,
                    }
                    for r in runs
                ]
        return []

    def compare_runs(self, run_ids: List[str], metrics: List[str] = None) -> Dict:
        """比较多个运行"""
        if not MLFLOW_AVAILABLE:
            return {}

        comparison = {}
        for run_id in run_ids:
            run = self._client.get_run(run_id)
            comparison[run_id] = {
                "params": run.data.params,
                "metrics": run.data.metrics if metrics is None else {
                    k: run.data.metrics.get(k) for k in metrics
                }
            }
        return comparison


# ============================================================
# 便捷函数
# ============================================================

_default_tracker: Optional[ExperimentTracker] = None


def get_tracker(
    experiment_name: str = "algvex_experiment",
    tracking_uri: str = "./mlruns",
) -> ExperimentTracker:
    """获取或创建默认追踪器"""
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = ExperimentTracker(experiment_name, tracking_uri)
    return _default_tracker


def log_backtest_result(
    tracker: ExperimentTracker,
    config: Dict,
    result: Dict,
    equity_curve: Any = None,
    trades: Any = None,
):
    """
    记录回测结果的便捷函数

    Args:
        tracker: 追踪器
        config: 回测配置
        result: 回测结果
        equity_curve: 权益曲线 DataFrame
        trades: 交易记录 DataFrame
    """
    # 记录配置参数
    tracker.log_params({
        "initial_capital": config.get("initial_capital"),
        "leverage": config.get("leverage"),
        "symbols": str(config.get("symbols")),
        "start_date": config.get("start_date"),
        "end_date": config.get("end_date"),
    })

    # 记录指标
    tracker.log_metrics({
        "total_return": result.get("total_return", 0),
        "sharpe_ratio": result.get("sharpe_ratio", 0),
        "max_drawdown": result.get("max_drawdown", 0),
        "win_rate": result.get("win_rate", 0),
        "profit_factor": result.get("profit_factor", 0),
        "total_trades": result.get("total_trades", 0),
    })

    # 记录权益曲线
    if equity_curve is not None:
        tracker.log_dataframe(equity_curve, "equity_curve.parquet")

    # 记录交易记录
    if trades is not None:
        tracker.log_dataframe(trades, "trades.parquet")
