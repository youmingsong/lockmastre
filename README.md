# myllama

一个单文件的本地大模型加载助手，自动扫描 Ollama / LM Studio 下载的模型，GUI 一键启动。

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Platform](https://img.shields.io/badge/Windows-11-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)


> 📸 <img width="1003" height="782" alt="image" src="https://github.com/user-attachments/assets/3a1d9204-d95f-45f9-a936-a2d19bba3a11" />
<img width="1003" height="782" alt="image" src="https://github.com/user-attachments/assets/9330d64f-8ad7-4fae-a0c6-0d86684dc742" />
<img width="1003" height="782" alt="image" src="https://github.com/user-attachments/assets/821495aa-7bc8-4d15-a4f5-ff62f4a9adb0" />

## ✨ 功能特性

- **桌面GUI**：原生 Windows 窗口，不是网页，双击即用
- **多显卡监控**：实时显示每张 NVIDIA 卡的占用率、温度、显存、功耗
- **模型自动扫描**：自动识别 LM Studio / Ollama 下载的 GGUF 模型
- **内置后端**：自带 llama.cpp CUDA 版（b11101），不用自己编译
- **多模态支持**：自动匹配 mmproj 视觉投影文件，支持 Qwen-VL 类模型
- **Chat 测试**：内置标签页直接对话，不用另开客户端
- **局域网访问**：监听 0.0.0.0:4444，手机/其他电脑也能调 API

## 📦 环境要求

- Python 3.10 及以上
- Windows 10 / 11
- （可选）NVIDIA 显卡 + CUDA 驱动，没有也能 CPU 跑

## 🚀 快速开始

```bash
# 克隆仓库
git clone https://github.com/youmingsong/myllama.git
cd myllama

# 安装依赖
pip install -r requirements.txt

# 启动
python myllama_server_gui.py
