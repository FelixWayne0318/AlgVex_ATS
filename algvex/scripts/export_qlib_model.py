#!/usr/bin/env python3
"""
Qlib 模型导出脚本

功能:
- 从 Qlib 训练的模型导出到生产格式
- 验证模型完整性
- 生成模型元数据
- 支持批量导出

用法:
    # 导出单个模型
    python scripts/export_qlib_model.py --model path/to/qlib_model.pkl --output models/exported/model_v1.pkl

    # 导出并指定特征
    python scripts/export_qlib_model.py --model model.pkl --output model_v1.pkl --features return_5m,atr_288

    # 验证已导出的模型
    python scripts/export_qlib_model.py --verify models/exported/model_v1.pkl

    # 列出所有已导出的模型
    python scripts/export_qlib_model.py --list

依赖:
    - algvex.research.qlib_adapter (仅用于导出)
    - algvex.production.model_loader (用于加载和验证)
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# 添加项目路径
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


def compute_model_hash(model: Any) -> str:
    """计算模型哈希"""
    try:
        model_bytes = pickle.dumps(model)
        hash_value = hashlib.sha256(model_bytes).hexdigest()[:16]
        return f"sha256:{hash_value}"
    except Exception:
        return "sha256:unknown"


def detect_model_type(model: Any) -> str:
    """检测模型类型"""
    type_name = type(model).__name__
    module_name = type(model).__module__

    if "lightgbm" in module_name.lower() or type_name == "Booster":
        return "lightgbm"
    elif "xgboost" in module_name.lower() or type_name == "Booster":
        return "xgboost"
    elif "sklearn" in module_name.lower():
        return f"sklearn.{type_name}"
    elif isinstance(model, dict):
        if "weights" in model:
            return "linear"
        return "dict"
    elif isinstance(model, np.ndarray):
        return "numpy"
    else:
        return type_name


def load_qlib_model(model_path: str) -> Any:
    """加载 Qlib 模型"""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Qlib 模型通常有 .model 属性
    if hasattr(model, "model"):
        return model.model, model
    else:
        return model, None


def export_model(
    model_path: str,
    output_path: str,
    features: Optional[List[str]] = None,
    metrics: Optional[Dict[str, float]] = None,
    model_id: Optional[str] = None,
    description: str = "",
) -> Dict[str, Any]:
    """
    导出 Qlib 模型到生产格式

    Args:
        model_path: Qlib 模型路径
        output_path: 输出路径
        features: 特征列表
        metrics: 评估指标
        model_id: 模型ID
        description: 模型描述

    Returns:
        导出结果信息
    """
    print(f"正在加载模型: {model_path}")
    model, qlib_wrapper = load_qlib_model(model_path)

    model_type = detect_model_type(model)
    model_hash = compute_model_hash(model)

    if model_id is None:
        model_id = Path(output_path).stem

    # 从 Qlib wrapper 提取信息 (如果有)
    if qlib_wrapper is not None:
        if features is None and hasattr(qlib_wrapper, "feature_names_"):
            features = qlib_wrapper.feature_names_

    # 构建元数据
    metadata = {
        "model_id": model_id,
        "model_type": model_type,
        "model_hash": model_hash,
        "features": features or [],
        "feature_count": len(features) if features else 0,
        "metrics": metrics or {},
        "description": description,
        "source": {
            "type": "qlib",
            "path": str(model_path),
        },
        "export_info": {
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "exported_by": "scripts/export_qlib_model.py",
            "format_version": "1.0",
        },
    }

    # 构建保存数据
    save_data = {
        "model": model,
        "metadata": metadata,
    }

    # 确保输出目录存在
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存
    print(f"正在导出模型到: {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(save_data, f)

    # 计算导出文件哈希
    with open(output_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    result = {
        "status": "success",
        "model_id": model_id,
        "model_type": model_type,
        "model_hash": model_hash,
        "file_hash": f"sha256:{file_hash}",
        "output_path": str(output_path),
        "file_size_bytes": output_path.stat().st_size,
        "features_count": len(features) if features else 0,
    }

    print(f"\n导出成功!")
    print(f"  模型ID: {model_id}")
    print(f"  模型类型: {model_type}")
    print(f"  模型哈希: {model_hash}")
    print(f"  文件哈希: sha256:{file_hash}")
    print(f"  文件大小: {result['file_size_bytes']:,} bytes")

    return result


def verify_model(model_path: str) -> Dict[str, Any]:
    """
    验证已导出的模型

    Args:
        model_path: 模型路径

    Returns:
        验证结果
    """
    model_path = Path(model_path)

    if not model_path.exists():
        return {
            "status": "error",
            "message": f"模型文件不存在: {model_path}",
        }

    try:
        with open(model_path, "rb") as f:
            data = pickle.load(f)

        # 检查格式
        if not isinstance(data, dict):
            return {
                "status": "warning",
                "message": "模型不是标准 AlgVex 格式 (非 dict)",
                "detected_type": type(data).__name__,
            }

        if "model" not in data:
            return {
                "status": "warning",
                "message": "模型缺少 'model' 键",
                "keys": list(data.keys()),
            }

        if "metadata" not in data:
            return {
                "status": "warning",
                "message": "模型缺少 'metadata' 键",
            }

        # 验证元数据
        metadata = data["metadata"]
        required_fields = ["model_id", "model_type", "features"]
        missing_fields = [f for f in required_fields if f not in metadata]

        if missing_fields:
            return {
                "status": "warning",
                "message": f"元数据缺少字段: {missing_fields}",
                "metadata": metadata,
            }

        # 重新计算哈希并验证
        model = data["model"]
        computed_hash = compute_model_hash(model)
        stored_hash = metadata.get("model_hash", "")

        hash_match = (computed_hash == stored_hash) if stored_hash else None

        # 计算文件哈希
        with open(model_path, "rb") as f:
            file_hash = f"sha256:{hashlib.sha256(f.read()).hexdigest()[:16]}"

        result = {
            "status": "valid",
            "model_id": metadata.get("model_id"),
            "model_type": metadata.get("model_type"),
            "features_count": len(metadata.get("features", [])),
            "stored_hash": stored_hash,
            "computed_hash": computed_hash,
            "hash_match": hash_match,
            "file_hash": file_hash,
            "file_size_bytes": model_path.stat().st_size,
            "exported_at": metadata.get("export_info", {}).get("exported_at"),
        }

        print(f"\n模型验证结果: {model_path}")
        print(f"  状态: {'有效' if hash_match else '待验证'}")
        print(f"  模型ID: {result['model_id']}")
        print(f"  模型类型: {result['model_type']}")
        print(f"  特征数量: {result['features_count']}")
        print(f"  哈希匹配: {hash_match}")
        print(f"  导出时间: {result['exported_at']}")

        return result

    except Exception as e:
        return {
            "status": "error",
            "message": f"验证失败: {str(e)}",
        }


def list_models(models_dir: str = "models/exported") -> List[Dict[str, Any]]:
    """
    列出所有已导出的模型

    Args:
        models_dir: 模型目录

    Returns:
        模型列表
    """
    models_dir = Path(models_dir)

    if not models_dir.exists():
        print(f"模型目录不存在: {models_dir}")
        return []

    models = []
    for ext in ["*.pkl", "*.pickle", "*.model"]:
        for model_path in models_dir.glob(ext):
            try:
                with open(model_path, "rb") as f:
                    data = pickle.load(f)

                if isinstance(data, dict) and "metadata" in data:
                    metadata = data["metadata"]
                    models.append({
                        "path": str(model_path),
                        "model_id": metadata.get("model_id"),
                        "model_type": metadata.get("model_type"),
                        "features_count": len(metadata.get("features", [])),
                        "exported_at": metadata.get("export_info", {}).get("exported_at"),
                        "file_size": model_path.stat().st_size,
                    })
                else:
                    models.append({
                        "path": str(model_path),
                        "model_id": model_path.stem,
                        "model_type": "unknown",
                        "features_count": 0,
                        "exported_at": None,
                        "file_size": model_path.stat().st_size,
                    })
            except Exception as e:
                models.append({
                    "path": str(model_path),
                    "model_id": model_path.stem,
                    "error": str(e),
                })

    # 输出表格
    if models:
        print(f"\n已导出的模型 ({models_dir}):")
        print("-" * 80)
        print(f"{'模型ID':<20} {'类型':<15} {'特征数':<8} {'大小':<12} {'导出时间'}")
        print("-" * 80)
        for m in models:
            if "error" in m:
                print(f"{m['model_id']:<20} {'ERROR':<15} {'-':<8} {'-':<12} {m['error']}")
            else:
                size = f"{m['file_size']:,} B"
                exported = m.get('exported_at', '-')[:19] if m.get('exported_at') else '-'
                print(f"{m['model_id']:<20} {m['model_type']:<15} {m['features_count']:<8} {size:<12} {exported}")
        print("-" * 80)
        print(f"共 {len(models)} 个模型")
    else:
        print(f"没有找到已导出的模型")

    return models


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="AlgVex Qlib 模型导出工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 导出模型
  python scripts/export_qlib_model.py --model qlib_lgb.pkl --output models/exported/lgb_v1.pkl

  # 导出并指定特征
  python scripts/export_qlib_model.py --model model.pkl --output out.pkl --features return_5m,atr_288,oi_change_rate

  # 验证模型
  python scripts/export_qlib_model.py --verify models/exported/lgb_v1.pkl

  # 列出所有模型
  python scripts/export_qlib_model.py --list
        """
    )

    # 导出选项
    parser.add_argument("--model", "-m", type=str, help="Qlib 模型路径")
    parser.add_argument("--output", "-o", type=str, help="输出路径")
    parser.add_argument("--features", "-f", type=str, help="特征列表 (逗号分隔)")
    parser.add_argument("--model-id", type=str, help="模型ID (默认使用文件名)")
    parser.add_argument("--description", "-d", type=str, default="", help="模型描述")

    # 验证选项
    parser.add_argument("--verify", "-v", type=str, help="验证已导出的模型")

    # 列出选项
    parser.add_argument("--list", "-l", action="store_true", help="列出所有已导出的模型")
    parser.add_argument("--models-dir", type=str, default="models/exported", help="模型目录")

    # 输出选项
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式")

    args = parser.parse_args()

    # 切换到项目根目录
    os.chdir(project_root)

    result = None

    if args.list:
        result = list_models(args.models_dir)

    elif args.verify:
        result = verify_model(args.verify)

    elif args.model and args.output:
        features = args.features.split(",") if args.features else None
        result = export_model(
            model_path=args.model,
            output_path=args.output,
            features=features,
            model_id=args.model_id,
            description=args.description,
        )

    else:
        parser.print_help()
        sys.exit(1)

    # JSON 输出
    if args.json and result:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
