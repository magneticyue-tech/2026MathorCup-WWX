# MathorCup 环境配置（Windows）

## 1. 安装软件
| 软件 | 下载 | 备注 |
|---|---|---|
| VS Code | https://code.visualstudio.com | 勾选”添加到 PATH” |
| Git | https://git-scm.com | 默认选项即可 |
| Miniconda | https://www.anaconda.com/download | 或更全面但更大的 Anaconda |
| TeX Live | https://tug.org/texlive/ | 选完整安装，约 8GB，耗时较长 |
| Node.js | https://nodejs.org | 选 LTS 版本，用于安装 Claude CLI |
| Codex (AI 编辑器) | https://codexeditor.com/download | 可选，AI 原生代码编辑器 |

VS Code 插件：`Python`、`Jupyter`、`LaTeX Workshop`、`Claude`

### Claude Code CLI 安装
```powershell
npm install -g @anthropic-ai/claude-code
claude --version
```

### Codex 安装说明
Codex 是独立的 AI 原生编辑器，安装后可直接使用，无需额外配置。与 VS Code 可并行使用。

## 2. Git 身份（首次）
```powershell
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

## 3. 获取项目并建环境
```powershell
git clone git@github.com:magneticyue-tech/mathorcup2026WWX.git mathorcup
cd mathorcup
conda env create -f environment.yml
conda activate mathorcup
python -c "import pandas, sklearn, xgboost, lightgbm; print('ok')"
```

**VS Code 选择解释器：**
- `Ctrl+Shift+P` → 输入 `Python: Select Interpreter` → 选择 `mathorcup`
- 或打开 `.py` 文件后点击右下角状态栏的解释器名称

下载中断（`IncompleteRead` / `Connection broken`）时，先删掉残缺环境，换清华镜像后重试：
```powershell
conda env remove -n mathorcup -y
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes
conda env create -f environment.yml
```

新增依赖后更新环境：
```powershell
conda env update -f environment.yml --prune
```

## 4. LaTeX 编译
论文在 `paper/main.tex`，首行 `% !TEX program = xelatex` 指定用 XeLaTeX（中文必须）。
- `Ctrl+Alt+B` 编译，`Ctrl+Alt+V` 预览
- 若仍调用 pdflatex 报 Unicode 错误：VS Code 设置搜 `latex-workshop.latex.recipe.default`，设为 `lastUsed` 或手动选 XeLaTeX recipe

## 5. 目录
```
data/raw        原始数据（不入 git）
data/processed  处理后数据（不入 git）
notebooks/      数据探索
src/            正式代码
results/        结果与图表
paper/          LaTeX 论文
```
数据文件较大，通过网盘共享，不要提交到仓库。
