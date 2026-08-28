# Sea-Ice Draft Reconstruction from 2D Imaging Sonar

[English](README.md) | [中文](README_CN.md)

> 更新日期：2026-08-28

**Version 1.0.0**

一个面向 Blueprint Subsea Oculus 二维成像声呐的研究代码参考实现，覆盖原始 `.oculus` 日志转 NetCDF 预处理，以及结合 AUV 深度和姿态数据的沿航迹海冰吃水重建。

This repository provides a research reference pipeline for reconstructing
along-track sea-ice draft from upward-looking 2D imaging sonar and synchronized
AUV motion data.

## 项目定位

本项目公开的是通用算法思路、处理流程和必要实现代码，不包含任何现场原始数据、导航记录、人工标注或正式科研结果。当前目录是独立整理的 GitHub 发布副本，不会读取或修改其他科研工作目录。

当前输入适配器重点面向 `oculus-python` 导出的 Oculus NetCDF 数据结构。其他二维成像声呐可以复用核心算法，但需要先将回波矩阵、时间及采样几何信息适配至 [`docs/INPUT_FORMAT.md`](docs/INPUT_FORMAT.md) 规定的接口。仓库对 `.oculus` 文件的解析依赖外部项目 [`oculus-python`](https://gitlab.gbar.dtu.dk/fletho/oculus-python)，并不包含该库的源码。

该方法完成的是二维成像声呐中心成像平面假设下的逐帧冰底代表线提取和沿航迹海冰吃水估算，不是完整的冰底三维重建，也不能在缺少干舷、雪深和静力平衡约束时给出海冰总厚度。

## Pipeline

```text
Oculus .oculus log
        ↓  oculus-python / bps_oculus_io
Oculus NetCDF polar sonar frames
        ↓
center-plane backward mapping
        ↓
range-adaptive background and artifact suppression
        ↓
candidate ice-band segmentation
        ↓
intensity-weighted representative boundary
        ↓
roll and pitch correction
        ↓
AUV depth − upward vertical range
        ↓
along-track sea-ice draft and QC tables
```

更完整的方法说明见 [`docs/METHOD.md`](docs/METHOD.md)。

## 目录结构

```text
imaging-sonar-sea-ice-draft/
├── ice_sonar_pipeline/       # 核心处理模块
├── config/                   # 通用示例配置
├── docs/                     # 方法、输入格式和发布检查
├── examples/                 # 仅生成合成测试数据
├── tests/                    # 几何与姿态校正测试
├── convert_oculus_to_netcdf.py # Oculus 原始日志转换入口
├── run_pipeline.py           # 命令行入口
├── run_from_config.py        # JSON 配置入口
├── requirements.txt
└── pyproject.toml
```

## 安装

建议使用 Python 3.10 或更高版本：

```bash
python -m venv .venv
```

Windows PowerShell（包含 Oculus 原始日志转换功能）：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[oculus]"
```

如果只处理已经准备好的 NetCDF，不需要安装 `oculus-python`：

```bash
python -m pip install -e .
```

`oculus-python 0.0.2a1` 当前为预发布版本；其中 v2 `.oculus` 日志支持由上游标记为 experimental。转换依赖额外约束 `numpy<2` 和 `opencv-python<4.12`，以保持当前依赖组合兼容，并避免修改第三方安装目录中的代码。

## 输入数据

海冰吃水重建需要两类用户自备数据：

1. 成像声呐 NetCDF：包含逐 ping 回波矩阵、时间和声呐采样几何信息；
2. AUV 运动 CSV：包含同步时间、深度以及经过质量控制的 roll、pitch 等字段。

字段要求见 [`docs/INPUT_FORMAT.md`](docs/INPUT_FORMAT.md)。当前读取器已适配 `oculus-python` 的 NetCDF 输出。不同设备导出的变量名或维度不同时，应先编写独立的数据适配器，不建议直接修改核心几何定义。

## Oculus 原始数据预处理

安装转换依赖后，可将单个 `.oculus` 日志导出为 NetCDF4：

```bash
python convert_oculus_to_netcdf.py \
  --input path/to/input.oculus \
  --output-dir path/to/netcdf_output
```

等价的已安装命令为：

```bash
oculus-to-netcdf --input path/to/input.oculus --output-dir path/to/netcdf_output
```

该包装器调用 `oculus-python` 提供的 `bps_oculus_io`，随后检查输出是否包含非空的 `backscatter` 变量。默认拒绝覆盖已有同名 NetCDF，并在转换完成后移除临时硬链接或副本。Windows 下如果第三方 HDF5/NetCDF 组件无法处理非 ASCII 路径，建议将输出目录设置为较短的纯英文路径。

## 直接运行

```bash
python run_pipeline.py \
  --nc-file path/to/sonar.nc \
  --motion-file path/to/auv_motion.csv \
  --output-root outputs/example \
  --timezone UTC \
  --ping-step 5 \
  --grid-range-mode metadata \
  --binary-tuning-workers 1
```

查看完整参数：

```bash
python run_pipeline.py --help
```

相对输出路径以当前工作目录为基准。默认使用单进程参数选择，以便获得更稳定、可复核的结果。

## 使用 JSON 配置

先复制并修改 [`config/example_run.json`](config/example_run.json)，再运行：

```bash
python run_from_config.py --config config/example_run.json
```

配置文件中的相对路径以配置文件所在目录为基准。

## 合成数据自检

仓库不附带真实观测数据。可以生成小型人工数据，检查安装、输入读取和基本运行流程：

```bash
python examples/generate_synthetic_inputs.py --output-dir examples/generated
python run_from_config.py --config config/example_run.json
```

合成数据仅用于软件冒烟测试，不应作为算法精度或极区适用性的验证依据。

## 人工掩膜

仓库不包含任何基于真实观测生成的人工掩膜。如果使用者有自己的 remove-only polygon mask，可通过 `--manual-mask-json` 显式提供。人工编辑必须在方法和结果中单独披露。

## 重要检查项

应用于新数据前，至少需要确认：

- 声呐距离、方位角和波束顺序；
- AUV/声呐坐标轴方向及 roll、pitch 符号；
- 声呐时间与导航时间是否同步；
- AUV 深度基准和传感器安装偏移；
- 默认阈值及形态约束是否适合新的量程和信噪比；
- 自动 QC 标记是否与人工抽查一致。

## 数据和隐私

`.gitignore` 默认排除 `.oculus`、NetCDF、MAT、HDF5、CSV、NumPy 数据、图像、动画、结果目录、人工掩膜、本地配置及压缩包。提交前仍应使用 `git status` 和内容检索再次检查，不能只依赖忽略规则。

## 引用、第三方致谢与许可证

与本算法相关的论文尚未发表，因此 Version 1.0.0 暂不提供论文引用格式；正式引用信息将在论文发表后补充。

本仓库原创代码版权归 Liyibo 个人所有，并依据 [`MIT License`](LICENSE) 发布。原始日志转换依赖 Apache-2.0 许可的 `oculus-python 0.0.2a1`；详细归属和致谢见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

## English summary

The workflow maps range–azimuth sonar frames onto a regular center-plane grid,
suppresses range-dependent background and structured artifacts, extracts an
intensity-weighted ice-bottom boundary, applies AUV attitude correction, and
calculates sea-ice draft from vehicle depth and upward vertical range. Users
must supply their own sonar and navigation data and validate coordinate,
timing, and installation conventions for each deployment.
