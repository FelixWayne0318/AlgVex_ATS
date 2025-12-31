"""
配置验证器测试
"""

import os
import tempfile
import pytest
import yaml

from algvex.shared.config_validator import (
    ConfigValidator,
    ConfigValidationError,
    validate_config_hash,
)


class TestConfigValidator:
    """配置验证器测试类"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = ConfigValidator(self.temp_dir)

    def teardown_method(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_compute_hash(self):
        """测试哈希计算"""
        config = {"key": "value", "number": 123}
        hash1 = ConfigValidator.compute_hash(config)

        # 相同内容应产生相同哈希
        hash2 = ConfigValidator.compute_hash(config)
        assert hash1 == hash2

        # 不同内容应产生不同哈希
        config2 = {"key": "different", "number": 456}
        hash3 = ConfigValidator.compute_hash(config2)
        assert hash1 != hash3

        # 哈希格式正确
        assert hash1.startswith("sha256:")

    def test_compute_hash_ignores_config_hash(self):
        """测试哈希计算忽略 config_hash 字段"""
        config1 = {"key": "value", "config_hash": "old_hash"}
        config2 = {"key": "value", "config_hash": "new_hash"}

        hash1 = ConfigValidator.compute_hash(config1)
        hash2 = ConfigValidator.compute_hash(config2)

        assert hash1 == hash2

    def test_load_config(self):
        """测试加载配置文件"""
        # 创建测试配置
        config = {
            "config_version": "1.0.0",
            "key": "value",
        }
        config_path = os.path.join(self.temp_dir, "test.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # 加载配置
        loaded = self.validator.load_config("test")
        assert loaded["config_version"] == "1.0.0"
        assert loaded["key"] == "value"

    def test_load_nonexistent_config(self):
        """测试加载不存在的配置"""
        with pytest.raises(ConfigValidationError):
            self.validator.load_config("nonexistent")

    def test_validate_hash(self):
        """测试哈希验证"""
        # 创建配置并设置正确的哈希
        config = {
            "config_version": "1.0.0",
            "key": "value",
        }
        config["config_hash"] = ConfigValidator.compute_hash(config)

        config_path = os.path.join(self.temp_dir, "valid.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # 验证应通过
        passed, message = self.validator.validate_hash("valid")
        assert passed

    def test_validate_all(self):
        """测试验证所有配置"""
        # 创建多个配置文件
        for name in ["config1", "config2"]:
            config = {
                "config_version": "1.0.0",
                "name": name,
            }
            config_path = os.path.join(self.temp_dir, f"{name}.yaml")
            with open(config_path, "w") as f:
                yaml.dump(config, f)

        # 验证所有
        results = self.validator.validate_all()
        assert len(results) == 2
        assert all(passed for passed, _ in results.values())


class TestVisibilityConfig:
    """可见性配置测试"""

    def test_visibility_yaml_structure(self):
        """测试可见性配置结构"""
        config_path = "algvex/config/visibility.yaml"
        if not os.path.exists(config_path):
            pytest.skip("配置文件不存在")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # 检查必要字段
        assert "config_version" in config
        assert "visibility_types" in config
        assert "source_visibility_map" in config

        # 检查可见性类型
        vis_types = config["visibility_types"]
        assert "realtime" in vis_types
        assert "bar_close" in vis_types
        assert "bar_close_delayed" in vis_types
        assert "scheduled" in vis_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
