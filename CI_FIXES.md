# GitHub Actions CI 修复指南

## 问题总结

当前仓库在推送代码时出现红色叉叉(CI 失败)的主要原因如下:

### 1. 缺少包安装步骤 (主要问题)

**文件**: `.github/workflows/python-tests.yml`

**问题**: 工作流尝试运行测试,但没有安装 `algvex` 包。测试文件中使用了 `from algvex.shared.xxx` 这样的导入,但包没有被安装到 Python 环境中。

**影响**: 所有需要导入 algvex 模块的测试都会失败,错误信息为:
```
ModuleNotFoundError: No module named 'algvex'
```

**解决方案**: 在运行测试之前添加包安装步骤

---

### 2. 工作目录与路径不一致

**文件**: `.github/workflows/python-tests.yml`

**问题**:
- 工作流执行 `cd algvex` 切换到 algvex 目录
- 但测试代码中使用 `algvex/config/visibility.yaml` 这样的路径
- 在 algvex 目录内,正确路径应该是 `config/visibility.yaml`

**影响**: 测试会因找不到配置文件而失败

---

### 3. Import 边界检查误报

**文件**: `.github/workflows/python-tests.yml` (第 140-153 行)

**问题**: 使用 `grep -r "import qlib"` 检查,如果没找到匹配(期望结果),grep 返回退出码 1,导致工作流失败。

**当前代码**:
```bash
if grep -r "import qlib" algvex/production/ 2>/dev/null; then
  echo "❌ production/ should not import qlib"
  exit 1
else
  echo "✅ production/ does not import qlib"
fi
```

**影响**: 即使代码正确(production 不导入 qlib),检查也可能失败

---

## 修复方案

### 方案 A: 推荐方案 - 从仓库根目录运行测试

这是最简单且最不容易出错的方案。

**修改 `.github/workflows/python-tests.yml`**:

#### 步骤 1: 添加包安装步骤

在第 51 行之后添加:

```yaml
      - name: Install algvex package
        run: |
          pip install -e .
```

#### 步骤 2: 修改测试运行方式

将所有 `cd algvex` 后的测试改为从根目录运行:

**修改前** (第 53-56 行):
```yaml
      - name: Run visibility tests
        run: |
          cd algvex
          python -m pytest tests/test_visibility_checker.py -v --tb=short
```

**修改后**:
```yaml
      - name: Run visibility tests
        run: |
          python -m pytest algvex/tests/test_visibility_checker.py -v --tb=short
```

类似地修改:
- Run config validator tests (第 58-65 行)
- Run factor engine tests (第 67-74 行)
- Run P0 tests (第 76-83 行)

#### 步骤 3: 修复 import 边界检查

**修改前** (第 136-155 行):
```yaml
      - name: Check import boundaries
        run: |
          echo "Checking import boundaries..."

          # 检查 production 目录不导入 qlib
          if grep -r "import qlib" algvex/production/ 2>/dev/null; then
            echo "❌ production/ should not import qlib"
            exit 1
          else
            echo "✅ production/ does not import qlib"
          fi

          # 检查 production 目录不导入 research
          if grep -r "from algvex.research" algvex/production/ 2>/dev/null; then
            echo "❌ production/ should not import from research/"
            exit 1
          else
            echo "✅ production/ does not import from research/"
          fi

          echo "✅ Import boundary check passed"
```

**修改后**:
```yaml
      - name: Check import boundaries
        run: |
          echo "Checking import boundaries..."

          # 检查 production 目录不导入 qlib
          if find algvex/production/ -name "*.py" -type f -exec grep -l "import qlib\|from qlib" {} \; | grep -q .; then
            echo "❌ production/ should not import qlib"
            find algvex/production/ -name "*.py" -type f -exec grep -l "import qlib\|from qlib" {} \;
            exit 1
          else
            echo "✅ production/ does not import qlib"
          fi

          # 检查 production 目录不导入 research
          if find algvex/production/ -name "*.py" -type f -exec grep -l "from algvex.research" {} \; | grep -q .; then
            echo "❌ production/ should not import from research/"
            find algvex/production/ -name "*.py" -type f -exec grep -l "from algvex.research" {} \;
            exit 1
          else
            echo "✅ production/ does not import from research/"
          fi

          echo "✅ Import boundary check passed"
```

---

### 方案 B: 备选方案 - 使用 PYTHONPATH

如果你更喜欢保持 `cd algvex` 的方式,可以设置 PYTHONPATH:

```yaml
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pytest pytest-cov pyyaml numpy pandas
          if [ -f algvex/requirements.txt ]; then
            pip install -r algvex/requirements.txt || true
          fi

      - name: Run visibility tests
        run: |
          export PYTHONPATH="${GITHUB_WORKSPACE}:${PYTHONPATH}"
          cd algvex
          python -m pytest tests/test_visibility_checker.py -v --tb=short
```

但这种方式仍需要修复测试文件中的配置文件路径问题。

---

## 快速修复清单

如果你想快速修复 CI,按照以下步骤:

### ✅ 已完成
- [x] 创建 `setup.py` 文件(已由 Claude 创建)

### 🔧 需要手动修改的文件

#### 1. `.github/workflows/python-tests.yml`

**在第 51 行后添加**:
```yaml
      - name: Install algvex package
        run: |
          pip install -e .
```

**修改第 53-83 行的所有测试步骤**,去掉 `cd algvex`,改为:
```yaml
      - name: Run visibility tests
        run: python -m pytest algvex/tests/test_visibility_checker.py -v --tb=short

      - name: Run config validator tests
        run: |
          if [ -f algvex/tests/test_config_validator.py ]; then
            python -m pytest algvex/tests/test_config_validator.py -v --tb=short
          else
            echo "⚠️ test_config_validator.py not found, skipping"
          fi

      - name: Run factor engine tests
        run: |
          if [ -f algvex/tests/test_factor_engine.py ]; then
            python -m pytest algvex/tests/test_factor_engine.py -v --tb=short
          else
            echo "⚠️ test_factor_engine.py not found, skipping"
          fi

      - name: Run P0 tests (critical path)
        run: |
          if [ -d algvex/tests/p0 ] && [ "$(ls -A algvex/tests/p0/*.py 2>/dev/null)" ]; then
            python -m pytest algvex/tests/p0/ -v --tb=short
          else
            echo "⚠️ No P0 tests found in algvex/tests/p0/, skipping"
          fi
```

**修改第 136-155 行的 import 边界检查**:
```yaml
      - name: Check import boundaries
        run: |
          echo "Checking import boundaries..."

          # 检查 production 目录不导入 qlib
          qlib_imports=$(find algvex/production/ -name "*.py" -type f -exec grep -l "import qlib\|from qlib" {} \; || true)
          if [ -n "$qlib_imports" ]; then
            echo "❌ production/ should not import qlib"
            echo "$qlib_imports"
            exit 1
          else
            echo "✅ production/ does not import qlib"
          fi

          # 检查 production 目录不导入 research
          research_imports=$(find algvex/production/ -name "*.py" -type f -exec grep -l "from algvex.research" {} \; || true)
          if [ -n "$research_imports" ]; then
            echo "❌ production/ should not import from research/"
            echo "$research_imports"
            exit 1
          else
            echo "✅ production/ does not import from research/"
          fi

          echo "✅ Import boundary check passed"
```

---

## 验证修复

修改完成后,在本地验证:

```bash
# 1. 安装包
pip install -e .

# 2. 运行测试
python -m pytest algvex/tests/test_visibility_checker.py -v
python -m pytest algvex/tests/test_config_validator.py -v
python -m pytest algvex/tests/test_factor_engine.py -v

# 3. 检查 import 边界
find algvex/production/ -name "*.py" -type f -exec grep -l "import qlib\|from qlib" {} \;
```

如果本地测试通过,推送到 GitHub 后 CI 应该会成功。

---

## 预期结果

修复后:
- ✅ `python-tests.yml` 工作流应该全部通过
- ✅ `ci.yml` 工作流应该继续正常工作
- ✅ 推送代码时不再出现红色叉叉

---

## 需要帮助?

如果在修改过程中遇到问题:
1. 检查 GitHub Actions 日志,查看具体错误信息
2. 确保 `setup.py` 文件在仓库根目录
3. 确保工作流文件的缩进正确(YAML 对缩进敏感)
4. 可以先在单个测试步骤上测试,确认可行后再应用到所有步骤

---

生成时间: 2026-01-01
生成工具: Claude Code
