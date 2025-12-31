# Qlib 小白入门指南（美股版）

> 适用对象：完全零基础的编程新手
> 操作系统：Windows 10/11
> 预计时间：30-60 分钟

---

## 目录

1. [第一步：安装 Python](#第一步安装-python)
2. [第二步：验证 Python 安装](#第二步验证-python-安装)
3. [第三步：创建项目文件夹](#第三步创建项目文件夹)
4. [第四步：创建虚拟环境](#第四步创建虚拟环境)
5. [第五步：安装 Qlib](#第五步安装-qlib)
6. [第六步：下载美股数据](#第六步下载美股数据)
7. [第七步：运行第一个示例](#第七步运行第一个示例)
8. [常见问题解决](#常见问题解决)

---

## 第一步：安装 Python

### 1.1 下载 Python

1. 打开浏览器（Chrome、Edge 等都可以）

2. 在地址栏输入以下网址，按回车：
   ```
   https://www.python.org/downloads/
   ```

3. 你会看到一个黄色的大按钮，写着 **"Download Python 3.x.x"**

   ![下载按钮示意](点击这个黄色按钮)

4. 点击这个黄色按钮，浏览器会开始下载一个文件，文件名类似：
   ```
   python-3.12.x-amd64.exe
   ```

5. 等待下载完成（大约 25MB，通常 1-2 分钟）

### 1.2 安装 Python

1. 找到下载的文件：
   - 通常在 `C:\Users\你的用户名\Downloads` 文件夹
   - 或者点击浏览器底部/右上角的下载图标找到它

2. **双击** 这个 `.exe` 文件运行安装程序

3. ⚠️ **重要！！！** 在安装界面的底部，你会看到一个复选框：
   ```
   ☐ Add python.exe to PATH
   ```
   **必须勾选这个选项！** 点击它变成 ☑️

4. 然后点击 **"Install Now"** 按钮（立即安装）

5. 如果弹出"是否允许此应用对你的设备进行更改"，点击 **"是"**

6. 等待安装进度条走完（大约 2-5 分钟）

7. 看到 **"Setup was successful"** 表示安装成功

8. 点击 **"Close"** 关闭安装程序

---

## 第二步：验证 Python 安装

### 2.1 打开命令提示符

1. 按键盘上的 **Windows 键**（键盘左下角，有 Windows 图标的键）

2. 直接输入：
   ```
   cmd
   ```

3. 你会看到搜索结果中出现 **"命令提示符"** 或 **"Command Prompt"**

4. 点击它打开（或者直接按回车）

5. 会弹出一个黑色的窗口，这就是命令提示符

### 2.2 验证 Python

1. 在黑色窗口中，输入以下内容（直接打字即可）：
   ```
   python --version
   ```

2. 按 **回车键**

3. 如果安装成功，你会看到类似这样的输出：
   ```
   Python 3.12.0
   ```
   （版本号可能略有不同，只要是 3.8 以上就可以）

4. 再输入以下内容验证 pip（Python 的包管理器）：
   ```
   pip --version
   ```

5. 按 **回车键**

6. 你应该看到类似这样的输出：
   ```
   pip 23.2.1 from C:\...\pip (python 3.12)
   ```

✅ 如果两个命令都有正确输出，恭喜你，Python 安装成功了！

❌ 如果显示 `'python' 不是内部或外部命令...`，请看文档最后的"常见问题解决"部分。

---

## 第三步：创建项目文件夹

### 3.1 创建文件夹

1. 打开 **文件资源管理器**（任务栏上的文件夹图标，或按 `Win + E`）

2. 在左侧点击 **"此电脑"** 或 **"This PC"**

3. 双击打开你想使用的盘符（如 **F盘** 或 **D盘**）

4. 在空白处 **右键单击**

5. 选择 **"新建"** → **"文件夹"**

6. 给文件夹命名为：
   ```
   qlib_project
   ```

7. 按 **回车键** 确认

现在你有了一个项目文件夹，例如：`F:\qlib_project`（以下示例使用 F 盘）

### 3.2 在命令提示符中进入这个文件夹

1. 回到刚才的命令提示符窗口（黑色窗口）
   - 如果已经关闭了，按照步骤 2.1 重新打开

2. 输入以下命令进入你的盘符（例如 F 盘）：
   ```
   F:
   ```

3. 按 **回车键**

4. 输入以下命令进入项目文件夹：
   ```
   cd qlib_project
   ```

5. 按 **回车键**

6. 现在你应该看到提示符变成了：
   ```
   F:\qlib_project>
   ```

---

## 第四步：创建虚拟环境

> 虚拟环境是一个独立的 Python 环境，可以避免不同项目之间的冲突

### 4.1 创建虚拟环境

1. 在命令提示符中，确保你在 `D:\qlib_project>` 目录下

2. 输入以下命令：
   ```
   python -m venv qlib_env
   ```

3. 按 **回车键**

4. 等待几秒钟，不会有任何输出，这是正常的

5. 命令执行完毕后，你会在 `D:\qlib_project` 文件夹里看到一个新的文件夹 `qlib_env`

### 4.2 激活虚拟环境

1. 输入以下命令激活虚拟环境：
   ```
   qlib_env\Scripts\activate
   ```

2. 按 **回车键**

3. 激活成功后，你会看到提示符变成了：
   ```
   (qlib_env) D:\qlib_project>
   ```

   注意前面多了 `(qlib_env)`，这表示虚拟环境已经激活了！

⚠️ **重要提示**：每次打开新的命令提示符窗口，都需要重新激活虚拟环境！

---

## 第五步：安装 Qlib

### 5.1 升级 pip

1. 确保虚拟环境已激活（提示符前面有 `(qlib_env)`）

2. 输入以下命令：
   ```
   python -m pip install --upgrade pip
   ```

3. 按 **回车键**

4. 等待升级完成（可能需要 1-2 分钟）

### 5.2 安装 Qlib

1. 输入以下命令安装 Qlib：
   ```
   pip install pyqlib
   ```

2. 按 **回车键**

3. 你会看到很多下载和安装信息滚动，这是正常的

4. 等待安装完成（可能需要 5-10 分钟，取决于网速）

5. 安装成功后，你会看到类似这样的信息：
   ```
   Successfully installed pyqlib-0.9.7 ...
   ```

### 5.3 验证 Qlib 安装

1. 输入以下命令：
   ```
   python -c "import qlib; print(qlib.__version__)"
   ```

2. 按 **回车键**

3. 如果显示版本号（如 `0.9.7`），说明安装成功！

---

## 第六步：下载美股数据

### 6.1 创建数据文件夹

1. 输入以下命令创建数据存放目录：
   ```
   mkdir data
   ```

2. 按 **回车键**

### 6.2 下载美股数据

1. 输入以下命令下载美股数据（**请将路径改成你的实际盘符**）：
   ```
   python -c "from qlib.tests.data import GetData; GetData().qlib_data(target_dir='F:/qlib_project/data/us_data', region='us')"
   ```

   **或者**创建一个下载脚本 `download_data.py`：
   ```python
   from qlib.tests.data import GetData
   # 请将路径改成你的实际盘符，如 F: 或 D:
   GetData().qlib_data(target_dir="F:/qlib_project/data/us_data", region="us")
   ```
   然后运行：`python download_data.py`

2. 按 **回车键**

3. 你会看到下载进度：
   ```
   Downloading data...
   [████████████████████████████████] 100%
   ```

4. 等待下载完成（美股数据约 2-3GB，可能需要 10-30 分钟）

5. 下载完成后，你会看到：
   ```
   Data downloaded successfully!
   ```

⚠️ **如果下载很慢或失败**，可以尝试使用镜像源，输入：
```
pip install baostock -i https://pypi.tuna.tsinghua.edu.cn/simple
```
然后手动下载数据（见常见问题部分）

---

## 第七步：运行第一个示例

### 7.1 创建 Python 脚本

1. 在文件资源管理器中，打开 `D:\qlib_project` 文件夹

2. 在空白处 **右键单击**

3. 选择 **"新建"** → **"文本文档"**

4. 把文件名改成：
   ```
   my_first_qlib.py
   ```

   ⚠️ 注意：文件后缀要从 `.txt` 改成 `.py`

   如果看不到文件后缀，在文件资源管理器顶部点击 **"查看"** → 勾选 **"文件扩展名"**

5. **右键单击** 这个 `.py` 文件

6. 选择 **"用记事本打开"** 或 **"Edit with Notepad"**

7. 在记事本中，**复制粘贴** 以下全部代码：

```python
"""
我的第一个 Qlib 程序 - 美股 LightGBM 模型示例
"""

import qlib
from qlib.constant import REG_US
from qlib.data.dataset import DatasetH
from qlib.contrib.data.handler import Alpha158

if __name__ == '__main__':
    print("=" * 50)
    print("欢迎使用 Qlib！")
    print("=" * 50)

    # 第1步：初始化 Qlib
    # ⚠️ 重要：请将下面的路径改成你的实际路径！
    print("\n[1/6] 正在初始化 Qlib...")
    qlib.init(
        provider_uri=r"F:\qlib_project\data\us_data",  # 数据路径（改成你的盘符）
        region=REG_US,
    )
    print("✓ Qlib 初始化完成！")

    # 第2步：选择股票（使用前100只，避免计算时间过长）
    print("\n[2/6] 正在选择股票...")
    from qlib.data import D
    all_instruments = D.list_instruments(D.instruments("all"), as_list=True)
    selected = all_instruments[:100]  # 只用前100只股票
    print(f"  已选择 {len(selected)} 只股票（共 {len(all_instruments)} 只）")

    # 第3步：准备数据（使用2018-2020年数据）
    print("\n[3/6] 正在准备数据...")
    handler = Alpha158(
        instruments=selected,
        start_time="2018-01-01",
        end_time="2020-11-01",
    )
    print("✓ 数据处理器创建完成！")

    # 第4步：创建数据集
    print("\n[4/6] 正在创建数据集...")
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": ("2018-01-01", "2019-06-30"),
            "valid": ("2019-07-01", "2019-12-31"),
            "test": ("2020-01-01", "2020-11-01"),
        }
    )
    print("✓ 数据集创建完成！")

    # 第5步：训练模型
    print("\n[5/6] 正在训练 LightGBM 模型...")
    from qlib.contrib.model.gbdt import LGBModel

    model = LGBModel(
        loss="mse",
        num_boost_round=100,
        early_stopping_rounds=20,
    )
    model.fit(dataset)
    print("✓ 模型训练完成！")

    # 第6步：预测
    print("\n[6/6] 正在进行预测...")
    predictions = model.predict(dataset, segment="test")
    print("✓ 预测完成！")

    # 显示结果
    print("\n" + "=" * 50)
    print("预测结果（前10条）:")
    print("=" * 50)
    print(predictions.head(10))

    predictions.to_csv("predictions.csv")
    print("\n✓ 已保存到 predictions.csv")
    print("\n恭喜！程序运行成功！")
```

8. 按 `Ctrl + S` 保存文件

9. 关闭记事本

### 7.2 运行程序

1. 回到命令提示符窗口

2. 确保虚拟环境已激活（提示符前有 `(qlib_env)`）

3. 确保在项目目录下（`D:\qlib_project>`）

4. 输入以下命令运行程序：
   ```
   python my_first_qlib.py
   ```

5. 按 **回车键**

6. 你会看到程序运行的输出：
   ```
   ==================================================
   欢迎使用 Qlib！这是你的第一个量化程序
   ==================================================

   [1/5] 正在初始化 Qlib...
   ✓ Qlib 初始化完成！

   [2/5] 正在准备数据...
   ...
   ```

7. 等待程序运行完成（可能需要 5-15 分钟，取决于电脑性能）

8. 程序结束后，你会在 `D:\qlib_project` 文件夹看到 `predictions.csv` 文件

🎉 **恭喜你！你已经成功运行了第一个量化投资程序！**

---

## 常见问题解决

### 问题1：`'python' 不是内部或外部命令`

**原因**：安装 Python 时没有勾选 "Add python.exe to PATH"

**解决方法**：
1. 重新运行 Python 安装程序
2. 选择 "Modify"（修改）
3. 勾选 "Add python.exe to PATH"
4. 完成安装后重新打开命令提示符

### 问题2：pip install pyqlib 报错 "No matching distribution found"

**原因**：Windows 上 pyqlib 需要 C++ 编译器，或 Python 版本不兼容

**解决方法（按顺序尝试）**：

**方法1：检查 Python 版本**
```
python --version
```
Qlib 支持 Python 3.8 - 3.11。如果是 3.12+，需要安装 3.11 版本。

**方法2：使用国内镜像**
```
pip install pyqlib -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**方法3：安装 Visual Studio Build Tools（推荐）**
1. 下载 Visual Studio Build Tools：
   ```
   https://visualstudio.microsoft.com/visual-cpp-build-tools/
   ```
2. 运行安装程序
3. 勾选 **"Desktop development with C++"**（使用 C++ 的桌面开发）
4. 安装完成后重启电脑
5. 重新运行：`pip install pyqlib`

**方法4：从源码安装**
```
pip install numpy cython
pip install --no-build-isolation git+https://github.com/microsoft/qlib.git
```

**方法5：使用 conda（最稳定）**
如果上述方法都失败，建议使用 Anaconda：
1. 下载安装 Anaconda：https://www.anaconda.com/download
2. 打开 Anaconda Prompt
3. 创建环境并安装：
   ```
   conda create -n qlib python=3.10
   conda activate qlib
   pip install pyqlib
   ```

### 问题3：pip 安装很慢或失败

**原因**：网络问题

**解决方法**：使用国内镜像源
```
pip install pyqlib -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题4：下载数据报错 "No module named 'qlib.run'"

**原因**：新版 Qlib 命令格式已更改

**解决方法**：使用新的下载命令（请替换成你的盘符）
```
python -c "from qlib.tests.data import GetData; GetData().qlib_data(target_dir='F:/qlib_project/data/us_data', region='us')"
```

### 问题5：下载数据失败

**原因**：网络不稳定

**解决方法**：
1. 多试几次
2. 使用 VPN
3. 或者手动下载数据（联系数据提供方）

### 问题6：内存不足

**原因**：数据量大，内存不够

**解决方法**：
1. 关闭其他程序
2. 使用更小的数据集（如只用 2 年数据）
3. 或者使用更小的股票池

### 问题7：运行报错 "does not contain data for day"

**错误信息**：
```
ValueError: instrument: {'__DEFAULT_FREQ': 'F:\\qlib_project\\data\\us_data'} does not contain data for day
```

**原因**：数据没有正确下载，或路径配置错误

**解决方法**：

1. **检查数据目录结构**：
   ```cmd
   dir F:\qlib_project\data\us_data
   ```
   应该包含：`calendars/`、`instruments/`、`features/` 三个文件夹

2. **如果目录为空，重新下载数据**：
   ```cmd
   python -c "from qlib.tests.data import GetData; GetData().qlib_data(target_dir='F:/qlib_project/data/us_data', region='us')"
   ```

3. **检查脚本中的路径是否正确**：
   确保 `my_first_qlib.py` 中的路径与你的实际数据位置一致：
   ```python
   qlib.init(
       provider_uri=r"F:\qlib_project\data\us_data",  # 必须与数据下载路径一致
       region=REG_US,
   )
   ```

### 问题8：运行报错 "freeze_support()" 或多进程错误

**错误信息**：
```
RuntimeError: An attempt has been made to start a new process before the
current process has finished its bootstrapping phase.
```

**原因**：Windows 多进程需要 `if __name__ == '__main__':` 保护

**解决方法**：
确保代码被 `if __name__ == '__main__':` 包裹：
```python
import qlib
# ... 其他 import

if __name__ == '__main__':
    # 所有执行代码放在这里
    qlib.init(...)
    # ...
```

### 问题9：程序运行很慢或卡住

**原因**：数据量太大（8000+股票 × 158因子）

**解决方法**：使用更少的股票（如100只）：
```python
from qlib.data import D
all_instruments = D.list_instruments(D.instruments("all"), as_list=True)
selected = all_instruments[:100]  # 只用前100只

handler = Alpha158(
    instruments=selected,  # 使用选定的股票
    ...
)
```

### 问题10：运行时出现其他红色错误信息

**解决方法**：
1. 截图错误信息
2. 在 GitHub Issues 搜索：https://github.com/microsoft/qlib/issues
3. 或者创建新 Issue 寻求帮助

---

## 下一步学习

恭喜你完成了入门！接下来可以：

### 推荐：使用 Jupyter Notebook（可视化学习）

Jupyter Notebook 是一个交互式编程环境，非常适合新手学习：
- 可以逐步执行代码，查看每一步的结果
- 支持可视化图表
- 方便调试和实验

**安装步骤：**

1. 确保虚拟环境已激活：
   ```cmd
   F:
   cd qlib_project
   qlib_env\Scripts\activate
   ```

2. 安装 Jupyter 和可视化库：
   ```cmd
   pip install jupyter matplotlib seaborn
   ```

3. 启动 Jupyter Notebook：
   ```cmd
   jupyter notebook
   ```

4. 浏览器会自动打开，你可以：
   - 点击右上角 **"New" → "Python 3"** 创建新 Notebook
   - 或者打开我们提供的教程文件 **`Qlib_Jupyter入门教程.ipynb`**

**配套教程文件：** `Qlib_Jupyter入门教程.ipynb`（包含完整的可视化示例）

### 其他学习方向

1. **学习更多模型**：
   - 在 `qlib-0.9.7/examples/benchmarks/` 目录下有很多示例

2. **学习回测**：
   - 了解如何评估策略表现

3. **学习策略开发**：
   - 开发自己的交易策略

4. **阅读官方文档**：
   - https://qlib.readthedocs.io/

---

## 每次使用的快速启动步骤

以后每次使用 Qlib，只需要：

1. 打开命令提示符（按 Win 键，输入 cmd，回车）

2. 进入项目目录（替换成你的盘符）：
   ```
   F:
   cd qlib_project
   ```

3. 激活虚拟环境：
   ```
   qlib_env\Scripts\activate
   ```

4. 运行你的程序：
   ```
   python my_first_qlib.py
   ```

---

> 文档版本：1.6
> 最后更新：2025年12月21日
> 适用 Qlib 版本：0.9.7
>
> 变更记录：
> - v1.6 (2025-12-21): 新增 Jupyter Notebook 教程，添加可视化学习指南
> - v1.5 (2025-01-21): 重写示例代码，使用100只股票+if __name__保护，解决性能和多进程问题
> - v1.4 (2025-01-21): 统一使用 F 盘作为默认示例，添加路径提示
> - v1.3 (2025-01-21): 修复问题编号重复，整理常见问题列表
> - v1.2 (2025-01): 修正数据下载命令，新增 "No module named qlib.run" 解决方案
> - v1.1 (2025-01): 新增 "pip install pyqlib 报错" 详细解决方案
> - v1.0 (2024-12): 初始版本
