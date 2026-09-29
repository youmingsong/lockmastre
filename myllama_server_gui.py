"""
My Llama Server GUI - 独立 llama-server 版
直连 llama.cpp 后端（路径由 backends.json 描述），不依赖 LM Studio/Ollama
"""
import sys
import os
import json
import re
import subprocess
import time
import urllib.request
import urllib.error
import zipfile
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QTextEdit, QSpinBox, QDoubleSpinBox,
    QGroupBox, QComboBox, QMessageBox, QTabWidget, QFileDialog, QCheckBox,
    QProgressBar
)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer


APP_DIR = os.path.dirname(os.path.abspath(__file__))
BACKENDS_JSON = os.path.join(APP_DIR, "backends.json")
CONFIG_PATH = os.path.join(APP_DIR, "gui_config.json")

# P1-1 后端描述（文件缺失/损坏时回退到此内置默认，数据与 backends.json 一致）
BACKENDS_DEFAULT = {
    "default": "llama-cuda",
    "mirrors": ["https://ghproxy.com/", "https://mirror.ghproxy.com/"],
    "download": {
        "repo": "ggml-org/llama.cpp",
        "asset_pattern": "cudart-llama-bin-win-cuda-12.4-x64.zip",
    },
    "backends": [
        {"id": "llama-cuda", "name": "llama.cpp CUDA（内置）",
         "binary": "llama-server/llama-server.exe",
         "health_path": "/health", "env_extra": {}, "args_overrides": []}
    ]
}


def load_backends():
    """读 backends.json；失败返回内置默认"""
    try:
        with open(BACKENDS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or not isinstance(data.get("backends"), list):
            return BACKENDS_DEFAULT
        if not any(b.get("id") and b.get("binary") for b in data["backends"]):
            return BACKENDS_DEFAULT
        return data
    except (OSError, json.JSONDecodeError):
        return BACKENDS_DEFAULT


def scan_gpus():
    """nvidia-smi → [{"id":0,"name":"NVIDIA ...","vram_gb":12}, ...]；无 GPU/无 nvidia-smi 返回 []"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout or ""
    except (OSError, subprocess.SubprocessError):
        out = ""
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                gpus.append({"id": int(parts[0]), "name": parts[1],
                             "vram_gb": round(float(parts[-1]) / 1024)})
            except ValueError:
                continue
    if gpus:
        return gpus
    # query 不可用（老驱动）→ 退回 -L：兼容 "[0] Name" 与 "GPU 0: Name" 两种行格式
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5).stdout or ""
    except (OSError, subprocess.SubprocessError):
        return []
    for line in out.splitlines():
        m = re.match(r"^\s*(?:\[(\d+)\]|GPU\s+(\d+)):\s+(.+?)\s*$", line)
        if not m:
            continue
        try:
            idx = int(m.group(1) or m.group(2))
        except (TypeError, ValueError):
            continue
        name = re.sub(r"\s*\(UUID:[^)]*\)\s*$", "", m.group(3)).strip()
        vm = re.search(r"(\d+(?:\.\d+)?)\s*GB\b", name, re.I)   # 名字里带显存的（如 P100-PCIE-16GB）
        gpus.append({"id": idx, "name": name,
                     "vram_gb": float(vm.group(1)) if vm else None})
    return gpus


VMM_PROJ_RE = re.compile(r"mmproj|vision|clip", re.I)


def scan_lmstudio(root):
    """扫 <root>/**/*.gguf。排除 mmproj/vision/clip 不当主模型；同目录同 stem 自动配对投影。
    返回 [{"display","path","mmproj"}]"""
    out = []
    if not os.path.isdir(root):
        return out
    by_dir = {}
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".gguf"):
                by_dir.setdefault(dirpath, []).append(fn)
    for d, files in by_dir.items():
        mains = [f for f in files if not VMM_PROJ_RE.search(f)]
        projs = [f for f in files if VMM_PROJ_RE.search(f)]
        for m in mains:
            full = os.path.join(d, m)
            rel = os.path.relpath(full, root).replace("\\", "/")
            stem = os.path.splitext(m)[0].lower()
            mm = ""
            for p in projs:
                pl = p.lower()
                if stem in pl or pl.replace("mmproj-", "").split("-")[0] in stem:
                    mm = os.path.join(d, p)
                    break
            out.append({"display": rel, "path": full, "mmproj": mm})
    out.sort(key=lambda x: x["display"].lower())
    return out


def scan_ollama(root):
    """扫 <root>/manifests/registry.ollama.ai/<owner>/<model>/<tag>。
    读 manifest JSON，取 size 最大 layer 作主模型 blob；mediaType 含 projector 的 layer 作 mmproj。
    返回 [{"display":"owner/model:tag","path":blob路径,"mmproj":...}]"""
    out = []
    manroot = os.path.join(root, "manifests", "registry.ollama.ai")
    blobroot = os.path.join(root, "blobs")
    if not os.path.isdir(manroot):
        return out
    for dirpath, _, files in os.walk(manroot):
        for fn in files:
            manpath = os.path.join(dirpath, fn)
            try:
                with open(manpath, "r", encoding="utf-8") as f:
                    man = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            layers = man.get("layers", [])
            if not layers:
                continue
            model_layer = max(layers, key=lambda l: l.get("size", 0))
            digest = model_layer.get("digest", "").replace("sha256:", "")
            model_blob = os.path.join(blobroot, f"sha256-{digest}")
            if not os.path.isfile(model_blob):
                continue
            mm = ""
            for l in layers:
                if "projector" in l.get("mediaType", ""):
                    d = l.get("digest", "").replace("sha256:", "")
                    cand = os.path.join(blobroot, f"sha256-{d}")
                    if os.path.isfile(cand):
                        mm = cand
                        break
            rel = os.path.relpath(manpath, manroot).replace("\\", "/")
            display = f"{rel.rsplit('/', 1)[0]}:{fn}" if "/" in rel else rel
            out.append({"display": display, "path": model_blob, "mmproj": mm})
    out.sort(key=lambda x: x["display"].lower())
    return out


def scan_all_models(extra_dirs=None):
    """合并默认目录（~/.lmstudio/models + ~/.ollama/models）+ 用户追加目录。按 path 去重。"""
    out = []
    home = os.path.expanduser("~")
    out.extend(scan_lmstudio(os.path.join(home, ".lmstudio", "models")))
    out.extend(scan_ollama(os.path.join(home, ".ollama", "models")))
    for d in (extra_dirs or []):
        if d and os.path.isdir(d):
            out.extend(scan_lmstudio(d))
    seen, uniq = set(), []
    for m in out:
        if m["path"] not in seen:
            seen.add(m["path"])
            uniq.append(m)
    return uniq

def is_even_split(text):
    """split 字符串是否长得像「自动均分」结果（恢复配置时判断要不要当作用户手改过）"""
    try:
        vals = [float(x) for x in text.replace(" ", "").split(",") if x != ""]
    except ValueError:
        return False
    return (len(vals) >= 2 and all(abs(v - vals[0]) < 1e-3 for v in vals)
            and abs(sum(vals) - 1.0) < 0.01)


# P1-4：从 llama-server 启动日志提取模型加载信息（规则按本机 build 11101 实测文案写，首跑后补全）
LOAD_INFO_RULES = [
    ("model file", r"loaded meta data with .* from (.+?) \(",
     lambda m: os.path.basename(m.group(1).strip('"'))),
    ("model size", r"file size\s*=\s*([\d.]+\s*[KMGT]i?B)",
     lambda m: m.group(1)),
    ("params", r"model params\s*=\s*([\d.]+\s*\w+)",
     lambda m: m.group(1)),
    ("GPU offload", r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers?\s+to GPU\b",
     lambda m: f"{m.group(1)}/{m.group(2)} layers"),
    ("context window", r"llama_context:\s*n_ctx\s*=\s*(\d+)",
     lambda m: m.group(1)),
    ("KV cache", r"KV buffer size\s*=\s*([\d.]+)\s*([KMGT]i?B)",
     lambda m: f"{m.group(1)} {m.group(2)}"),
    ("flash attention", r"flash_attn\s*=\s*(\S+)",
     lambda m: m.group(1)),
]
LOAD_INFO_KEYS = [k for k, _, _ in LOAD_INFO_RULES] + ["load time"]


def extract_load_info(line):
    """单行日志 → {字段: 值}；无命中返回 {}"""
    out = {}
    for key, pattern, fmt in LOAD_INFO_RULES:
        m = re.search(pattern, line)
        if m:
            out[key] = fmt(m)
    return out


def _num(x):
    """宽松转 float；'N/A'/空/非数字 → None"""
    try:
        return float(str(x).strip())
    except (TypeError, ValueError):
        return None


class ServerThread(QThread):
    log = pyqtSignal(str)

    def __init__(self, cmd, env):
        super().__init__()
        self.cmd = cmd
        self.env = env
        self.cwd = os.path.dirname(os.path.abspath(cmd[0]))
        self.proc = None
        self.exit_rc = None

    def run(self):
        self.log.emit("$ " + " ".join(map(str, self.cmd)))
        self.proc = subprocess.Popen(
            self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=self.env,
            cwd=self.cwd,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        )
        for line in self.proc.stdout:
            self.log.emit(line.rstrip())
        self.exit_rc = self.proc.wait()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class ChatThread(QThread):
    token = pyqtSignal(str)
    done = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, base_url, messages, temperature=0.3):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.messages = messages
        self.temperature = temperature
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        try:
            payload = json.dumps({
                "messages": self.messages,
                "stream": True,
                "temperature": self.temperature
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                buf = b""
                while not self._stopped:
                    chunk = resp.read(65536) or b""
                    buf += chunk
                    idx = buf.find(b"\n")
                    while idx != -1:
                        line = buf[:idx].decode("utf-8", errors="replace").strip()
                        buf = buf[idx + 1:]
                        if line.startswith("data: ") and line[6:].strip():
                            data = line[6:].strip()
                            if data == "[DONE]":
                                self.done.emit()
                                return
                            try:
                                delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                                if delta:
                                    self.token.emit(delta)
                            except (json.JSONDecodeError, KeyError, IndexError):
                                pass
                        idx = buf.find(b"\n")
                    if not chunk:
                        break
            if not self._stopped:
                self.done.emit()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:300]
            self.error.emit(f"HTTP {e.code}: {body}")
        except urllib.error.URLError as e:
            self.error.emit(f"连接失败: {getattr(e, 'reason', str(e))}")
        except Exception as e:
            if not self._stopped:
                self.error.emit(str(e))


class HealthCheckThread(QThread):
    """启动后轮询 /health，模型加载完（status=ok）才 emit ready。不设硬超时：大模型加载可能超过提示阈值，只 warn 不 fail"""
    ready = pyqtSignal()
    warned_once = pyqtSignal(str)

    def __init__(self, base_url, timeout_s=60, interval_s=0.5):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.deadline = time.monotonic() + timeout_s
        self.interval_s = interval_s
        self.timeout_s = timeout_s
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        warned = False
        while not self._stopped:
            if self._poll_once():
                self.ready.emit()
                return
            if not warned and time.monotonic() >= self.deadline:
                # 不终止轮询：大模型加载可能超过提示阈值，服务起来后照常置绿；只报一次
                self.warned_once.emit(f"{self.timeout_s}s 未就绪（继续等待中，大模型加载较慢属正常，可检查日志）")
                warned = True
            for _ in range(int(self.interval_s * 10)):
                if self._stopped:
                    return
                time.sleep(0.1)

    def _poll_once(self):
        """返回 True=就绪。注意 llama-server 加载期间 /health 可能 200 但 status=loading，必须解析 body"""
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as r:
                body = r.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError, ConnectionError):
            return False
        try:
            status = json.loads(body).get("status")
        except json.JSONDecodeError:
            # 旧 build /health 可能非 JSON：能连通且有响应视为就绪
            return True
        return status == "ok"


class SelfTestThread(QThread):
    """就绪后自动发最小请求验证端到端（P2-4）"""
    result = pyqtSignal(bool, str)

    def __init__(self, base_url):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        payload = json.dumps({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read().decode("utf-8", errors="replace")
            data = json.loads(body)
            if data.get("choices"):
                self.result.emit(True, "端到端自检通过")
            else:
                self.result.emit(False, f"端到端自检失败: 响应无 choices: {body[:200]}")
        except (urllib.error.URLError, OSError, ValueError) as e:
            self.result.emit(False, f"端到端自检失败: {e}")


class DownloadThread(QThread):
    """P1-2：自动下载 llama-server 预编译包。
    流程：查 GitHub releases 找目标 asset → 按 直连/mirrors 顺序下载 → 校验大小 → 剥顶层目录解压 → 写 VERSION。
    零额外依赖（urllib + zipfile + json）。"""
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)        # done_bytes, total_bytes
    finished_ok = pyqtSignal(str)           # version tag e.g. "b11224"
    failed = pyqtSignal(str)

    def __init__(self, dest_dir, repo, asset_pattern, mirrors):
        super().__init__()
        self.dest_dir = dest_dir
        self.repo = repo
        self.asset_pattern = asset_pattern
        self.mirrors = mirrors or []
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        tmp_path = os.path.join(self.dest_dir, ".download.tmp")
        try:
            os.makedirs(self.dest_dir, exist_ok=True)
            tag, name, url, size = self._find_asset()
            self.log.emit(f"找到 {name}（约 {size/1024/1024:.0f} MB），版本 {tag}")

            candidates = [("直连 GitHub", url)]
            for i, m in enumerate(self.mirrors, 1):
                candidates.append((f"镜像{i} {m.rstrip('/')}/", m.rstrip("/") + "/" + url))

            ok = False
            for label, cand in candidates:
                if self._stopped:
                    return
                self.log.emit(f"尝试下载源：{label}")
                try:
                    self._download_one(cand, tmp_path, size)
                    ok = True
                    self.log.emit(f"  ✓ {label} 下载完成")
                    break
                except Exception as e:
                    self.log.emit(f"  ✗ {label} 失败：{e}")
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
            if not ok:
                self.failed.emit("所有下载源均失败。请检查网络；或手动把 zip 解压到 llama-server/ 目录后重启。")
                return

            self.log.emit("解压中...")
            self._extract_strip_top(tmp_path, self.dest_dir)
            try:
                os.remove(tmp_path)
            except OSError:
                pass

            with open(os.path.join(self.dest_dir, "VERSION"), "w", encoding="utf-8") as f:
                f.write(tag)
            self.log.emit(f"✅ 安装完成：{tag}")
            self.finished_ok.emit(tag)
        except Exception as e:
            self.failed.emit(f"下载失败：{e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _find_asset(self):
        """遍历最近 releases，找第一个 assets 里含 asset_pattern 的；返回 (tag, name, url, size)"""
        api = f"https://api.github.com/repos/{self.repo}/releases?per_page=20"
        req = urllib.request.Request(api, headers={"User-Agent": "myllama-gui"})
        with urllib.request.urlopen(req, timeout=15) as r:
            releases = json.loads(r.read().decode("utf-8"))
        for rel in releases:
            for a in rel.get("assets", []):
                if a.get("name") == self.asset_pattern:
                    return rel["tag_name"], a["name"], a["browser_download_url"], a["size"]
        raise RuntimeError(f"最近 20 个 release 里没找到资产 {self.asset_pattern}")

    def _download_one(self, url, tmp_path, expected_size):
        req = urllib.request.Request(url, headers={"User-Agent": "myllama-gui"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            total = int(resp.headers.get("Content-Length") or expected_size or 0)
            done = 0
            with open(tmp_path, "wb") as f:
                while not self._stopped:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        self.progress.emit(done, total)
        if self._stopped:
            raise RuntimeError("已取消")
        actual = os.path.getsize(tmp_path)
        if expected_size and actual != expected_size:
            raise RuntimeError(f"大小不符：期望 {expected_size}，实际 {actual}")

    @staticmethod
    def _extract_strip_top(zip_path, dest_dir):
        """解压 zip，剥掉顶层目录（zip 里第一层是 cudart-llama-bin-win-cuda-.../，内容直接摊到 dest_dir）"""
        with zipfile.ZipFile(zip_path) as z:
            for member in z.infolist():
                parts = member.filename.split("/", 1)
                if len(parts) < 2 or not parts[0]:
                    continue   # zip 根目录下散文件（罕见），跳过
                rel = parts[1]
                if not rel:
                    continue
                target = os.path.join(dest_dir, rel.replace("/", os.sep))
                if member.is_dir():
                    os.makedirs(target, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with z.open(member) as src, open(target, "wb") as dst:
                        while True:
                            buf = src.read(65536)
                            if not buf:
                                break
                            dst.write(buf)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MYLLAMA GUI (修车佬的本地模型助手)")
        self.resize(1000, 750)
        self.server_thread = None
        self.chat_thread = None
        self.health_thread = None
        self.selftest_thread = None
        self.dl_thread = None
        self.chat_history = []
        self.current_reply = ""
        # P1-1 后端可插拔：实际路径由 backends.json 描述，GUI 下拉选择
        self.backends_data = load_backends()
        self._split_manual = False   # tensor-split 被手改过 → 不再自动覆盖
        self._ready_flag = False
        self._manual_stop = False
        self.gpu_timer = None
        self._gpu_status_rows = {}
        self._gpu_status_fail = 0
        self.txt_loadinfo = None
        self._load_info = {}
        self._server_start = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        tabs = QTabWidget()
        root.addWidget(tabs)

        # === Tab 1: 启动（最常用：模型/网络/按钮/GPU状态/日志） ===
        tab_launch = QWidget()
        sl = QVBoxLayout(tab_launch)
        tabs.addTab(tab_launch, "启动")
        # === Tab 2: 参数（不常改的高级项） ===
        tab_params = QWidget()
        slp = QVBoxLayout(tab_params)
        tabs.addTab(tab_params, "参数")

        # GPU 状态实时卡片（启动 tab 顶部）
        self.gpu_list = scan_gpus()
        if self.gpu_list:
            sl.addWidget(self._build_gpu_status_card())

        # 后端
        gb_backend = QGroupBox("后端")
        blay = QVBoxLayout(gb_backend)
        row_be = QHBoxLayout()
        self.cb_backend = QComboBox()
        row_be.addWidget(self.cb_backend, 1)
        self.lbl_backend_status = QLabel("")
        row_be.addWidget(self.lbl_backend_status)
        self.btn_download = QPushButton("下载")
        self.btn_download.clicked.connect(self.start_download)
        row_be.addWidget(self.btn_download)
        blay.addLayout(row_be)
        self.prog_dl = QProgressBar()
        self.prog_dl.setVisible(False)
        blay.addWidget(self.prog_dl)
        for b in self.backends_data.get("backends", []):
            self.cb_backend.addItem(b["name"], b["id"])
        self.cb_backend.setCurrentIndex(0)
        self.cb_backend.currentIndexChanged.connect(lambda _: self._refresh_backend_status())
        sl.addWidget(gb_backend)

        # 模型
        gb = QGroupBox("模型（自动扫描 LM Studio / Ollama 目录）")
        lay = QVBoxLayout(gb)
        row_m = QHBoxLayout()
        self.cb_model = QComboBox()
        self.cb_model.setEditable(True)
        self.cb_model.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.cb_model.lineEdit().setPlaceholderText("选择或输入模型文件路径")
        self.cb_model.currentIndexChanged.connect(self._on_model_changed)
        row_m.addWidget(self.cb_model, 1)
        btn = QPushButton("浏览...")
        btn.clicked.connect(self.pick_model)
        row_m.addWidget(btn)
        lay.addLayout(row_m)
        row_mm = QHBoxLayout()
        row_mm.addWidget(QLabel("视觉投影 --mmproj"))
        self.ed_mmproj = QLineEdit()
        self.ed_mmproj.setPlaceholderText("可选：多模态模型视觉投影（自动配对，可手动改）")
        row_mm.addWidget(self.ed_mmproj, 1)
        btn_mm = QPushButton("...")
        btn_mm.setFixedWidth(30)
        btn_mm.clicked.connect(self.pick_mmproj)
        row_mm.addWidget(btn_mm)
        lay.addLayout(row_mm)
        sl.addWidget(gb)

        # 网络
        gb = QGroupBox("网络")
        lay = QHBoxLayout(gb)
        lay.addWidget(QLabel("监听 --host"))
        self.cb_host = QComboBox()
        self.cb_host.setEditable(True)
        self.cb_host.addItems(["127.0.0.1", "0.0.0.0"])
        self.cb_host.setCurrentText("127.0.0.1")
        lay.addWidget(self.cb_host)
        self.chk_lan = QCheckBox("允许局域网访问（监听 0.0.0.0）")
        lay.addWidget(self.chk_lan)
        self.chk_lan.toggled.connect(self._on_lan_toggled)
        self.cb_host.currentTextChanged.connect(self._sync_lan_from_host)
        lay.addWidget(QLabel("端口 --port"))
        self.ed_port = QLineEdit("4444")
        lay.addWidget(self.ed_port)
        sl.addWidget(gb)

        # 按钮
        lay_btn = QHBoxLayout()
        self.btn_start = QPushButton("启动服务")
        self.btn_start.clicked.connect(self.start_server)
        self.btn_stop = QPushButton("停止服务")
        self.btn_stop.clicked.connect(self.stop_server)
        self.btn_stop.setEnabled(False)
        lay_btn.addWidget(self.btn_start)
        lay_btn.addWidget(self.btn_stop)
        sl.addLayout(lay_btn)

        # 模型加载信息
        gb_load = QGroupBox("模型加载信息")
        self.txt_loadinfo = QTextEdit()
        self.txt_loadinfo.setReadOnly(True)
        self.txt_loadinfo.setStyleSheet("font-family: Consolas, 'Courier New', monospace; background:#f7f7f7; color:#222;")
        lb = QVBoxLayout(gb_load)
        lb.addWidget(self.txt_loadinfo)
        sl.addWidget(gb_load)

        # 日志
        sl.addWidget(QLabel("日志"))
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        sl.addWidget(self.txt_log, 1)

        # ====== Tab 2: 参数 ======
        # 推理参数
        gb = QGroupBox("推理参数")
        lay = QVBoxLayout(gb)
        lay_row1 = QHBoxLayout()
        lay_row1.addWidget(QLabel("上下文 -c"))
        self.cb_ctx = QComboBox()
        self._ctx_map = {"8K": 8192, "16K": 16384, "32K": 32768,
                         "64K": 65536, "128K": 131072, "256K": 262144}
        self.cb_ctx.addItems(list(self._ctx_map.keys()))
        self.cb_ctx.setCurrentText("32K")
        lay_row1.addWidget(self.cb_ctx)
        lay_row1.addWidget(QLabel("GPU层数 -ngl"))
        self.sp_ngl = QSpinBox()
        self.sp_ngl.setRange(-1, 999)
        self.sp_ngl.setSpecialValueText("自动")
        self.sp_ngl.setValue(-1)
        self.sp_ngl.setToolTip("-1=全部offload到GPU（自动）")
        lay_row1.addWidget(self.sp_ngl)
        lay_row1.addWidget(QLabel("线程 -t"))
        self.sp_threads = QSpinBox()
        self.sp_threads.setRange(1, 128)
        self.sp_threads.setValue(8)
        lay_row1.addWidget(self.sp_threads)
        lay.addLayout(lay_row1)
        lay_row2 = QHBoxLayout()
        lay_row2.addWidget(QLabel("温度 --temp"))
        self.dsb_srv_temp = QDoubleSpinBox()
        self.dsb_srv_temp.setRange(0.0, 2.0)
        self.dsb_srv_temp.setSingleStep(0.05)
        self.dsb_srv_temp.setValue(0.3)
        self.dsb_srv_temp.setToolTip("代码模型建议0.1-0.3")
        lay_row2.addWidget(self.dsb_srv_temp)
        lay_row2.addWidget(QLabel("重复惩罚 --repeat-penalty"))
        self.dsb_repeat = QDoubleSpinBox()
        self.dsb_repeat.setRange(1.0, 2.0)
        self.dsb_repeat.setSingleStep(0.05)
        self.dsb_repeat.setValue(1.1)
        lay_row2.addWidget(self.dsb_repeat)
        lay_row2.addWidget(QLabel("Top-P --top-p"))
        self.dsb_topp = QDoubleSpinBox()
        self.dsb_topp.setRange(0.0, 1.0)
        self.dsb_topp.setSingleStep(0.05)
        self.dsb_topp.setValue(0.9)
        lay_row2.addWidget(self.dsb_topp)
        lay_row2.addWidget(QLabel("Top-K --top-k"))
        self.sp_topk = QSpinBox()
        self.sp_topk.setRange(1, 100)
        self.sp_topk.setValue(40)
        lay_row2.addWidget(self.sp_topk)
        lay.addLayout(lay_row2)
        lay_row3 = QHBoxLayout()
        lay_row3.addWidget(QLabel("风格预设"))
        self._presets = {
            "严谨": (0.1, 1.0, 0.9, 40),
            "均衡": (0.7, 1.1, 0.9, 40),
            "急速": (0.3, 1.0, 0.95, 80),
            "癫狂": (1.0, 1.0, 0.95, 100),
            "长文": (0.5, 1.2, 0.9, 40),
        }
        for name, (t, r, p, k) in self._presets.items():
            btn = QPushButton(name)
            btn.setFixedWidth(50)
            btn.clicked.connect(lambda checked, v=(t,r,p,k): self._apply_preset(v))
            lay_row3.addWidget(btn)
        lay_row3.addStretch(1)
        lay.addLayout(lay_row3)
        slp.addWidget(gb)

        # 多卡
        self.gb_gpu = QGroupBox("多卡")
        self.gb_gpu.setToolTip("勾选的卡决定 CUDA_VISIBLE_DEVICES；更改需重启服务才生效")
        gpu_lay = QVBoxLayout(self.gb_gpu)
        g1 = QHBoxLayout()
        g1.addWidget(QLabel("使用GPU"))
        self._chk_gpus = {}
        if self.gpu_list:
            for g in self.gpu_list:
                label = f"[{g['id']}] {g['name']}"
                if g.get("vram_gb"):
                    label += f" ({g['vram_gb']:g}GB)"
                chk = QCheckBox(label)
                chk.setChecked(True)
                chk.toggled.connect(self._on_gpu_check_changed)
                self._chk_gpus[g["id"]] = chk
                g1.addWidget(chk)
        else:
            g1.addWidget(QLabel("未检测到 NVIDIA GPU（CPU 模式或无 nvidia-smi）"))
        g1.addStretch(1)
        gpu_lay.addLayout(g1)
        g2 = QHBoxLayout()
        g2.addWidget(QLabel("tensor-split"))
        self.ed_split = QLineEdit("")
        self.ed_split.setToolTip("各卡比例，如 0.5,0.5；改勾选自动均分（手改后以手动为准）")
        self.ed_split.textEdited.connect(self._on_split_edited)
        g2.addWidget(self.ed_split, 1)
        g2.addWidget(QLabel("-sm"))
        self.cb_sm = QComboBox()
        self.cb_sm.addItems(["row", "layer"])
        g2.addWidget(self.cb_sm)
        gpu_lay.addLayout(g2)
        if not self.gpu_list:
            self.gb_gpu.setVisible(False)
        slp.addWidget(self.gb_gpu)

        # 加速 / 显存
        gb = QGroupBox("加速 / 显存")
        lay = QHBoxLayout(gb)
        self.chk_fa = QCheckBox("Flash Attention")
        self.chk_fa.setChecked(True)
        self.chk_kv = QCheckBox("KV缓存 q4_0")
        self.chk_kv.setChecked(True)
        lay.addWidget(self.chk_fa)
        lay.addWidget(self.chk_kv)
        slp.addWidget(gb)

        # 投机解码
        gb = QGroupBox("投机解码（需 MTP/草稿模型支持；改完重启服务）")
        lay = QHBoxLayout(gb)
        self.chk_spec = QCheckBox("启用")
        lay.addWidget(self.chk_spec)
        lay.addWidget(QLabel("类型"))
        self.cb_spec = QComboBox()
        self.cb_spec.addItems(["draft-mtp", "draft-simple", "draft-eagle3", "ngram-mod"])
        self.cb_spec.setToolTip("draft-mtp: DeepSeek V3 等带 MTP 模块的模型；ngram-mod: 任意模型免草稿加速")
        lay.addWidget(self.cb_spec)
        lay.addWidget(QLabel("n-max"))
        self.sp_spec_nmax = QSpinBox()
        self.sp_spec_nmax.setRange(1, 16)
        self.sp_spec_nmax.setValue(3)
        lay.addWidget(self.sp_spec_nmax)
        lay.addWidget(QLabel("草稿设备"))
        self.cb_spec_dev = QComboBox()
        self.cb_spec_dev.addItem("跟主模型", "")
        for g in self.gpu_list:
            self.cb_spec_dev.addItem(f"CUDA{g['id']} ({g['name'][:24]})", f"CUDA{g['id']}")
        self.cb_spec_dev.addItem("CPU（不 offload）", "none")
        self.cb_spec_dev.setToolTip("草稿模型单独放哪张卡；不选=跟主模型同卡")
        lay.addWidget(self.cb_spec_dev)
        lay.addWidget(QLabel("草稿ngl"))
        self.sp_spec_ngld = QSpinBox()
        self.sp_spec_ngld.setRange(-1, 999)
        self.sp_spec_ngld.setSpecialValueText("auto")
        self.sp_spec_ngld.setValue(-1)
        lay.addWidget(self.sp_spec_ngld)
        lay.addStretch(1)
        row_draft = QHBoxLayout()
        row_draft.addWidget(QLabel("草稿模型文件"))
        self.ed_draft_model = QLineEdit()
        self.ed_draft_model.setPlaceholderText("draft-simple/eagle3 用：选小草稿模型 gguf（draft-mtp/ngram 留空）")
        row_draft.addWidget(self.ed_draft_model, 1)
        btn_draft = QPushButton("...")
        btn_draft.setFixedWidth(30)
        btn_draft.clicked.connect(self.pick_draft_model)
        row_draft.addWidget(btn_draft)
        lay.addLayout(row_draft)
        slp.addWidget(gb)

        # 思考模式
        gb = QGroupBox("思考模式（R1/Qwen3-thinking 等；改完重启服务）")
        lay = QHBoxLayout(gb)
        lay.addWidget(QLabel("reasoning"))
        self.cb_reasoning = QComboBox()
        self.cb_reasoning.addItems(["auto", "on", "off"])
        self.cb_reasoning.setToolTip("auto=模型自决；on/off 强制开关")
        lay.addWidget(self.cb_reasoning)
        lay.addWidget(QLabel("effort"))
        self.ed_effort = QLineEdit("default")
        self.ed_effort.setMaximumWidth(100)
        self.ed_effort.setPlaceholderText("default")
        lay.addWidget(self.ed_effort)
        lay.addWidget(QLabel("budget(-1=不限)"))
        self.sp_budget = QSpinBox()
        self.sp_budget.setRange(-1, 200000)
        self.sp_budget.setValue(-1)
        lay.addWidget(self.sp_budget)
        self.chk_rformat = QCheckBox("reasoning_format=deepseek（外部客户端可见思考链）")
        lay.addWidget(self.chk_rformat)
        lay.addStretch(1)
        slp.addWidget(gb)
        slp.addStretch(1)

        # === Tab 2: Chat ===
        tab2 = QWidget()
        cl = QVBoxLayout(tab2)
        tabs.addTab(tab2, "Chat 测试")
        self.lbl_status = QLabel("⚪ 服务未启动")
        self.lbl_status.setStyleSheet("padding:4px; font-size:13px;")
        cl.addWidget(self.lbl_status)
        row_url = QHBoxLayout()
        self.ed_baseurl = QLineEdit()
        self.ed_baseurl.setReadOnly(True)
        row_url.addWidget(self.ed_baseurl, 1)
        btn_copy = QPushButton("复制")
        btn_copy.clicked.connect(self._copy_baseurl)
        row_url.addWidget(btn_copy)
        cl.addLayout(row_url)
        cl.addWidget(QLabel("对话窗口"))
        self.txt_chat = QTextEdit()
        self.txt_chat.setReadOnly(True)
        cl.addWidget(self.txt_chat, 1)
        lay_in = QHBoxLayout()
        self.ed_input = QLineEdit()
        self.ed_input.setPlaceholderText("输入消息，回车发送...")
        self.ed_input.returnPressed.connect(self.send_chat)
        lay_in.addWidget(self.ed_input, 1)
        lay_in.addWidget(QLabel("温度"))
        self.dsb_temp = QDoubleSpinBox()
        self.dsb_temp.setRange(0.0, 2.0)
        self.dsb_temp.setSingleStep(0.1)
        self.dsb_temp.setValue(0.3)
        self.dsb_temp.setToolTip("代码模型建议0.1-0.3；对话建议0.6-0.8")
        lay_in.addWidget(self.dsb_temp)
        btn_send = QPushButton("发送")
        btn_send.clicked.connect(self.send_chat)
        lay_in.addWidget(btn_send)
        cl.addLayout(lay_in)
        btn_clear = QPushButton("清空对话")
        btn_clear.clicked.connect(lambda: (self.txt_chat.clear(), setattr(self, 'chat_history', [])))
        cl.addWidget(btn_clear)

        self.load_config()
        self._refresh_backend_status()

    def _refresh_backend_status(self):
        """检测后端 binary 是否在位，更新状态标签和下载按钮文案"""
        binary = self.resolve_backend()
        if binary and os.path.isfile(binary):
            ver = ""
            vf = os.path.join(APP_DIR, "llama-server", "VERSION")
            try:
                with open(vf, "r", encoding="utf-8") as f:
                    ver = f.read().strip()
            except OSError:
                pass
            self.lbl_backend_status.setText(f"✅ 就绪{f'（{ver}）' if ver else ''}")
            self.lbl_backend_status.setStyleSheet("color:#1a7f37;")
            if not (self.dl_thread and self.dl_thread.isRunning()):
                self.btn_download.setText("重新下载")
        else:
            self.lbl_backend_status.setText("⚠️ 缺少 llama-server，点右侧下载")
            self.lbl_backend_status.setStyleSheet("color:#b41217; font-weight:bold;")
            if not (self.dl_thread and self.dl_thread.isRunning()):
                self.btn_download.setText("下载")

    def start_download(self):
        """点「下载/取消」按钮：未下载时启动 DownloadThread，下载中则取消"""
        if self.dl_thread and self.dl_thread.isRunning():
            self.dl_thread.stop()
            self.log("下载取消中...")
            return
        dl_cfg = self.backends_data.get("download") or BACKENDS_DEFAULT.get("download") or {}
        if not dl_cfg:
            QMessageBox.warning(self, "未配置下载", "backends.json 缺少 download 段")
            return
        entry = self.current_backend_entry()
        dest_dir = os.path.join(APP_DIR, os.path.dirname(entry["binary"]))
        self.btn_download.setText("取消")
        self.prog_dl.setVisible(True)
        self.prog_dl.setValue(0)
        self.log(f"===== 开始下载 {dl_cfg['asset_pattern']} =====")
        self.dl_thread = DownloadThread(
            dest_dir=dest_dir,
            repo=dl_cfg["repo"],
            asset_pattern=dl_cfg["asset_pattern"],
            mirrors=self.backends_data.get("mirrors", []),
        )
        self.dl_thread.log.connect(self.log)
        self.dl_thread.progress.connect(
            lambda d, t: self.prog_dl.setValue(int(d * 100 / t) if t else 0))
        self.dl_thread.finished_ok.connect(self._on_download_done)
        self.dl_thread.failed.connect(self._on_download_failed)
        self.dl_thread.start()

    def _on_download_done(self, version):
        self.btn_download.setText("重新下载")
        self.prog_dl.setVisible(False)
        self._refresh_backend_status()

    def _on_download_failed(self, msg):
        self.log(f"❌ {msg}")
        self.btn_download.setText("重试下载")
        self.prog_dl.setVisible(False)
        self._refresh_backend_status()

    def load_config(self):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                c = json.load(f)
            history = c.get("model_history", [])
            for h in history:
                self.cb_model.addItem(h, h)   # display=path, userData=path（兼容旧历史）
            self._scanned_models = scan_all_models(c.get("scan_dirs", []))
            seen_paths = set(self.cb_model.itemData(i) for i in range(self.cb_model.count()))
            for m in self._scanned_models:
                if m["path"] not in seen_paths:
                    self.cb_model.addItem(m["display"], m["path"])
                    seen_paths.add(m["path"])
            cur_model = c.get("model", "")
            ctx_val = c.get("ctx", "32K")
            if isinstance(ctx_val, int):
                # 旧配置存的是数字，自动匹配最近档位
                ctx_val = min(self._ctx_map.items(), key=lambda x: abs(x[1] - ctx_val))[0]
            self.cb_ctx.setCurrentText(ctx_val)
            self.sp_ngl.setValue(c.get("ngl", -1))
            self.sp_threads.setValue(c.get("threads", 8))
            host = c.get("host", "127.0.0.1")
            if "lan_access" not in c and host == "0.0.0.0":
                # 旧配置（无 lan_access 字段）：行为保持不变，只提示一次
                self.log("注意: 当前配置允许局域网访问 (host=0.0.0.0)，仅本机使用请取消勾选「允许局域网访问」")
            self.cb_host.setCurrentText(host)   # _sync_lan_from_host 会按 host 对齐勾选框状态
            # P1-1：新字段 backend；旧版 gui_config.json 的自定义 backend_bin 已弃用（只提示一次）
            if not c.get("backend") and c.get("backend_bin"):
                self.log(f"注意: 旧配置自定义后端路径 {c['backend_bin']} 已被弃用，请从「后端」下拉重新选择（见 backends.json）")
            bid = c.get("backend")
            if bid:
                idx = self.cb_backend.findData(bid)
                if idx >= 0:
                    self.cb_backend.setCurrentIndex(idx)
            else:
                self.cb_backend.setCurrentIndex(0)
            self.ed_port.setText(c.get("port", "8080"))
            # GPU 选择：优先新字段 gpu_ids；兼容旧版 gpus 字符串（"0,1"）；一个都没勾=用全部卡
            ids = c.get("gpu_ids")
            if ids is None:
                ids = [int(x) for x in re.findall(r"\d+", str(c.get("gpus", "") or ""))]
            hit = [i for i in (ids or []) if i in self._chk_gpus]
            for gid, chk in self._chk_gpus.items():
                chk.setChecked(bool(hit) and gid in hit)
            # tensor-split：均分视为自动生成的结果；非均分说明用户手改过 → 不再覆盖
            self.ed_split.setText(str(c.get("split", "")))
            self._split_manual = bool(self.ed_split.text().strip()) \
                and not is_even_split(self.ed_split.text())
            self.cb_sm.setCurrentText(c.get("sm", "row"))
            self.chk_fa.setChecked(c.get("flash_attn", True))
            self.chk_kv.setChecked(c.get("kv_q4", True))
            self.chk_spec.setChecked(c.get("spec_enable", False))
            self.cb_spec.setCurrentText(c.get("spec_type", "draft-mtp"))
            self.sp_spec_nmax.setValue(c.get("spec_nmax", 3))
            dev = c.get("spec_device", "")
            idx = self.cb_spec_dev.findData(dev)
            if idx >= 0:
                self.cb_spec_dev.setCurrentIndex(idx)
            self.sp_spec_ngld.setValue(c.get("spec_ngld", -1))
            self.ed_draft_model.setText(c.get("draft_model", ""))
            self.cb_reasoning.setCurrentText(c.get("reasoning", "auto"))
            self.ed_effort.setText(c.get("reasoning_effort", "default"))
            self.sp_budget.setValue(c.get("reasoning_budget", -1))
            self.chk_rformat.setChecked(c.get("reasoning_format") == "deepseek")
            self.dsb_temp.setValue(c.get("temperature", 0.3))
            self.dsb_srv_temp.setValue(c.get("srv_temp", 0.3))
            self.dsb_repeat.setValue(c.get("repeat_penalty", 1.1))
            self.dsb_topp.setValue(c.get("top_p", 0.9))
            self.sp_topk.setValue(c.get("top_k", 40))
            idx = self.cb_model.findData(cur_model)
            if idx >= 0:
                self.cb_model.setCurrentIndex(idx)
            elif cur_model:
                self.cb_model.setCurrentText(cur_model)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

    def save_config(self):
        c = {
            "model": self._current_model_path(),
            "models": [{"display": self.cb_model.itemText(i), "path": self.cb_model.itemData(i) or self.cb_model.itemText(i)}
                       for i in range(self.cb_model.count())],
            "scan_dirs": [],   # TODO M6 追加目录

            "ctx": self.cb_ctx.currentText(),
            "ngl": self.sp_ngl.value(),
            "threads": self.sp_threads.value(),
            "host": self.cb_host.currentText(),
            "lan_access": self.chk_lan.isChecked(),
            "backend": self.cb_backend.currentData(),
            "port": self.ed_port.text(),
            "gpu_ids": self._selected_gpu_ids(),
            "split": self.ed_split.text(),
            "sm": self.cb_sm.currentText(),
            "flash_attn": self.chk_fa.isChecked(),
            "kv_q4": self.chk_kv.isChecked(),
            "spec_enable": self.chk_spec.isChecked(),
            "spec_type": self.cb_spec.currentText(),
            "spec_nmax": self.sp_spec_nmax.value(),
            "spec_device": self.cb_spec_dev.currentData(),
            "spec_ngld": self.sp_spec_ngld.value(),
            "draft_model": self.ed_draft_model.text(),
            "reasoning": self.cb_reasoning.currentText(),
            "reasoning_effort": self.ed_effort.text(),
            "reasoning_budget": self.sp_budget.value(),
            "reasoning_format": "deepseek" if self.chk_rformat.isChecked() else "none",
            "temperature": self.dsb_temp.value(),
            "srv_temp": self.dsb_srv_temp.value(),
            "repeat_penalty": self.dsb_repeat.value(),
            "top_p": self.dsb_topp.value(),
            "top_k": self.sp_topk.value(),
        }
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(c, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _apply_preset(self, vals):
        t, r, p, k = vals
        self.dsb_srv_temp.setValue(t)
        self.dsb_repeat.setValue(r)
        self.dsb_topp.setValue(p)
        self.sp_topk.setValue(k)

    def _on_lan_toggled(self, checked):
        """局域网勾选框 → 写 host（blockSignals 防止与 _sync_lan_from_host 互触发）"""
        self.cb_host.blockSignals(True)
        self.cb_host.setCurrentText("0.0.0.0" if checked else "127.0.0.1")
        self.cb_host.blockSignals(False)

    def _sync_lan_from_host(self, text):
        """host 被手改时对齐勾选框状态（仅 0.0.0.0 视为局域网访问）"""
        checked = text.strip() == "0.0.0.0"
        if checked != self.chk_lan.isChecked():
            self.chk_lan.blockSignals(True)
            self.chk_lan.setChecked(checked)
            self.chk_lan.blockSignals(False)

    def current_backend_entry(self):
        entries = self.backends_data.get("backends") or BACKENDS_DEFAULT["backends"]
        target_id = self.cb_backend.currentData()
        for b in entries:
            if b.get("id") == target_id:
                return b
        return entries[0]

    def resolve_backend(self):
        """选中后端的可执行文件绝对路径；不存在返回 None"""
        entry = self.current_backend_entry()
        p = os.path.join(APP_DIR, entry["binary"])
        return p if os.path.isfile(p) else None

    def build_cmd(self, backend, params):
        """P1-1：按后端 + 当前控件值拼 llama.cpp 命令行。
        不做通用模板引擎（只有一个后端时是过度设计）；--mmproj 透传、args_overrides 追加尾参。"""
        cmd = [self.resolve_backend(),
               "-m", params["model"],
               "-lv", "4"]   # 加载信息卡片依赖详细日志
        cmd += ["-c", str(params["ctx"]), "-t", str(params["threads"])]
        cmd += ["--temp", str(params["srv_temp"]), "--repeat-penalty", str(params["repeat_penalty"]),
                "--top-p", str(params["top_p"]), "--top-k", str(params["top_k"])]
        cmd += ["--host", params["host"], "--port", params["port"], "-sm", params["sm"]]
        ngl = params.get("ngl", -1)
        split = (params.get("split") or "").strip()
        ids = params.get("gpu_ids") or []
        if ngl >= 0:
            cmd += ["-ngl", str(ngl)]
            if split and len(ids) >= 2:
                cmd += ["--tensor-split", split]
        if params.get("flash_attn"):
            cmd += ["--flash-attn", "on"]
        if params.get("kv_q4"):
            cmd += ["--cache-type-k", "q4_0", "--cache-type-v", "q4_0"]
        mmproj = (params.get("mmproj") or "").strip()
        if mmproj:
            cmd += ["--mmproj", mmproj]
        # P1-6 投机解码：勾选后追加 --spec-type + n-max
        if params.get("spec_enable"):
            st = params.get("spec_type", "draft-mtp")
            cmd += ["--spec-type", st]
            nmax = params.get("spec_nmax", 3)
            if nmax and nmax > 0:
                cmd += ["--spec-draft-n-max", str(nmax)]
            dev = params.get("spec_device") or ""
            if dev:
                cmd += ["--device-draft", dev]
            ngld = params.get("spec_ngld", -1)
            if ngld is not None and ngld >= 0:
                cmd += ["--ngld", str(ngld)]
            draft = (params.get("draft_model") or "").strip()
            if draft:
                cmd += ["--model-draft", draft]
        # P1-7 思考开关：--reasoning 恒传（auto 对非思考模型无害）；effort/budget 非默认才加
        reason = params.get("reasoning", "auto")
        if reason and reason != "auto":
            cmd += ["--reasoning", reason]
        effort = (params.get("reasoning_effort") or "").strip()
        if effort and effort != "default":
            cmd += ["--reasoning-effort", effort]
        budget = params.get("reasoning_budget", -1)
        if budget is not None and budget >= 0:
            cmd += ["--reasoning-budget", str(budget)]
        if params.get("reasoning_format") == "deepseek":
            cmd += ["--reasoning-format", "deepseek"]
        cmd.extend([a for a in backend.get("args_overrides") or []])
        return cmd

    def gpu_env(self):
        """带 CUDA_VISIBLE_DEVICES 的 env；仅当用户取消勾选了部分卡时才限制可见卡"""
        env = os.environ.copy()
        all_ids = [g["id"] for g in self.gpu_list]
        if self.gpu_list and sorted(self._selected_gpu_ids()) != sorted(all_ids):
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in self._selected_gpu_ids())
        return env

    def _on_gpu_check_changed(self, _checked=False):
        self._refresh_split()

    def _on_split_edited(self, _text=""):
        self._split_manual = True

    def _selected_gpu_ids(self):
        """当前勾选的卡号；一个都没勾=使用全部"""
        ids = sorted(gid for gid, chk in self._chk_gpus.items() if chk.isChecked())
        return ids or [g["id"] for g in self.gpu_list]

    def _refresh_split(self):
        """勾选变化时自动均分 tensor-split（用户手改过则不覆盖）"""
        if self._split_manual:
            return
        n = len(self._selected_gpu_ids())
        if n < 2:
            text = ""
        else:
            step = round(1.0 / n, 4)
            vals = [step] * (n - 1) + [round(1.0 - step * (n - 1), 4)]
            text = ",".join(f"{v:.4f}".rstrip("0").rstrip(".") for v in vals)
        self.ed_split.setText(text)

    # ---------- P1-3 GPU 状态实时卡片 ----------

    def _build_gpu_status_card(self):
        """每秒由 QTimer 刷新的 GPU 状态组：每卡一行（名称/利用率条/温度·显存·功耗）"""
        gb = QGroupBox("GPU 状态")
        lay = QVBoxLayout(gb)
        for g in self.gpu_list:
            row = QHBoxLayout()
            name_lbl = QLabel(f"[{g['id']}] {g['name']}")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFixedWidth(200)
            info_lbl = QLabel("")
            row.addWidget(name_lbl)
            row.addWidget(bar)
            row.addWidget(info_lbl)
            self._gpu_status_rows[g["id"]] = {"label": name_lbl, "bar": bar, "info": info_lbl}
            lay.addLayout(row)
        self.gpu_timer = QTimer()
        self.gpu_timer.setInterval(1000)
        self.gpu_timer.timeout.connect(self._refresh_gpu_status)
        self.gpu_timer.start()
        return gb

    @staticmethod
    def _temp_color(temp):
        if temp is None:
            return ""
        if temp < 60:
            return "#1a7f37"
        if temp < 80:
            return "#c58a00"
        return "#c0392b"

    def _refresh_gpu_status(self):
        """nvidia-smi 查询温度/利用率/显存/功耗并刷新每卡一行"""
        out = None
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,temperature.gpu,utilization.gpu,"
                 "memory.used,memory.total,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2).stdout or ""
        except Exception:
            pass
        parsed = {}
        if out:
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 7:
                    continue
                try:
                    idx = int(parts[0])
                except ValueError:
                    continue
                parsed[idx] = {
                    "temp": _num(parts[2]), "util": _num(parts[3]),
                    "used": _num(parts[4]), "total": _num(parts[5]),
                    "power": _num(parts[6])}
        if not parsed:
            self._gpu_status_fail += 1
            if self._gpu_status_fail >= 3:
                for r in self._gpu_status_rows.values():
                    r["info"].setText("传感器不可用（已停止刷新）")
                    r["bar"].setValue(0)
                self.gpu_timer.stop()
            return
        self._gpu_status_fail = 0
        for g in self.gpu_list:
            d = parsed.get(g["id"])
            if d is None:
                continue
            r = self._gpu_status_rows[g["id"]]
            util = d["util"] or 0.0
            r["bar"].setValue(int(max(0, min(100, util))))
            temp_s = f"{d['temp']:g}" if d["temp"] is not None else "N/A"
            used = (d["used"] / 1024) if d["used"] is not None else 0.0
            total = (d["total"] / 1024) if d["total"] is not None else 0.0
            power_s = f"{d['power']:g} W" if d["power"] is not None else ""
            info = f"{temp_s}\u00b0C  VRAM {used:.1f}/{total:.1f} GB" + (f"  {power_s}" if power_s else "")
            r["info"].setText(info)
            color = self._temp_color(d["temp"])
            if color:
                r["info"].setStyleSheet(f"color:{color};")

    # ---------- P1-4 模型加载信息卡片 ----------

    def _on_server_log(self, line):
        """服务日志行 → 滚动日志 + 按规则更新加载信息卡片"""
        self.log(line)
        if not hasattr(self, "txt_loadinfo") or self.txt_loadinfo is None:
            return
        if self._server_start is None:
            return
        for key, val in extract_load_info(line).items():
            self._load_info[key] = val
        if "listening" in line and "load time" not in self._load_info:
            self._load_info["load time"] = f"{time.monotonic() - self._server_start:.1f}s"
        self._render_load_card()

    def _render_load_card(self):
        lines = []
        for key in LOAD_INFO_KEYS:
            v = self._load_info.get(key)
            if v:
                lines.append(f"{key}: {v}")
        if self.txt_loadinfo:
            self.txt_loadinfo.setPlainText("\n".join(lines))

    def log(self, msg):
        self.txt_log.append(msg)

    def _current_model_path(self):
        """当前选中模型的真实路径：优先 userData（扫描到的），否则 lineEdit 手输的文本"""
        d = self.cb_model.currentData()
        if d:
            return d
        return self.cb_model.currentText().strip()

    def _on_model_changed(self, _idx=-1):
        """切模型时：自动带出配对 mmproj + 按文件名套参数预设"""
        path = self._current_model_path()
        if not path:
            return
        for m in getattr(self, "_scanned_models", []):
            if m["path"] == path:
                self.ed_mmproj.setText(m.get("mmproj") or "")
                break
        base = os.path.basename(path).lower()
        if "coder" in base or "code" in base:
            self.cb_ctx.setCurrentText("32K")
            self.dsb_srv_temp.setValue(0.3)
            self.log(f"↳ 检测到代码模型，自动套用 32K / temp 0.3")

    def pick_mmproj(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视觉投影文件", "", "GGUF 投影 (*.gguf);;所有文件 (*)"
        )
        if path:
            self.ed_mmproj.setText(path)

    def pick_draft_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择草稿模型文件（draft-simple/eagle3 用）", "", "草稿模型 (*.gguf);;所有文件 (*)"
        )
        if path:
            self.ed_draft_model.setText(path)

    def pick_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "模型文件 (*.gguf *.bin *.safetensors);;所有文件 (*)"
        )
        if path:
            self._add_model_history(path)
            self.cb_model.setCurrentText(path)

    def _add_model_history(self, path):
        """把路径加到下拉历史，去重，最多保留20条"""
        path = path.strip()
        if not path:
            return
        # 移除已有的相同项
        for i in range(self.cb_model.count()):
            if self.cb_model.itemText(i) == path:
                self.cb_model.removeItem(i)
                break
        # 插到最前面
        self.cb_model.insertItem(0, path)
        # 限制最多20条
        while self.cb_model.count() > 20:
            self.cb_model.removeItem(self.cb_model.count() - 1)

    def start_server(self):
        if self.server_thread and self.server_thread.isRunning():
            return
        model = self._current_model_path().strip()
        if not model or not os.path.isfile(model):
            QMessageBox.critical(self, "错误", "请选择存在的模型文件")
            return
        entry = self.current_backend_entry()
        backend_bin = self.resolve_backend()
        if not backend_bin or not os.path.isfile(backend_bin):
            self.log(f"❌ 找不到后端 {entry['name']}（{entry.get('binary','')}），请点上方「下载」按钮获取 llama-server")
            self._refresh_backend_status()
            return

        # 启动时把模型加入历史
        self._add_model_history(model)

        host = self.cb_host.currentText().strip()
        port = self.ed_port.text().strip()
        ids = self._selected_gpu_ids()   # 勾选的卡；一个都没勾=用全部卡

        params = {
            "model": model,
            "ctx": str(self._ctx_map[self.cb_ctx.currentText()]),
            "threads": self.sp_threads.value(),
            "srv_temp": self.dsb_srv_temp.value(),
            "repeat_penalty": self.dsb_repeat.value(),
            "top_p": self.dsb_topp.value(),
            "top_k": self.sp_topk.value(),
            "host": host,
            "port": port,
            "sm": self.cb_sm.currentText(),
            "ngl": self.sp_ngl.value(),
            "flash_attn": self.chk_fa.isChecked(),
            "kv_q4": self.chk_kv.isChecked(),
            "split": self.ed_split.text(),
            "gpu_ids": ids,
            "mmproj": self.ed_mmproj.text(),   # P1-5：视觉投影（自动配对或手动选）
            "spec_enable": self.chk_spec.isChecked(),
            "spec_type": self.cb_spec.currentText(),
            "spec_nmax": self.sp_spec_nmax.value(),
            "spec_device": self.cb_spec_dev.currentData(),
            "spec_ngld": self.sp_spec_ngld.value(),
            "draft_model": self.ed_draft_model.text(),
            "reasoning": self.cb_reasoning.currentText(),
            "reasoning_effort": self.ed_effort.text(),
            "reasoning_budget": self.sp_budget.value(),
            "reasoning_format": "deepseek" if self.chk_rformat.isChecked() else "none",
        }
        cmd = self.build_cmd(entry, params)

        split = (params["split"] or "").strip()
        n_parts = len([x for x in split.split(",") if x.strip()])
        if split and n_parts != len(ids):
            self.log(f"⚠️ tensor-split 有 {n_parts} 段但选了 {len(ids)} 张卡，比例可能错位，请检查")

        env = self.gpu_env()
        for k, v in entry.get("env_extra", {}).items():
            env[k] = str(v)

        self._server_start = time.monotonic()
        self._load_info = {}
        if hasattr(self, "txt_loadinfo") and self.txt_loadinfo:
            self.txt_loadinfo.clear()

        self._manual_stop = False
        self.log("===== 启动 llama-server =====")
        if env.get("CUDA_VISIBLE_DEVICES"):
            self.log(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
        self.server_thread = ServerThread(cmd, env)
        self.server_thread.log.connect(self._on_server_log)
        self.server_thread.finished.connect(lambda: self.on_stopped(host, port))
        self.server_thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.gb_gpu.setEnabled(False)   # llama.cpp 限制：GPU 分配变更需重启服务，运行中整组禁用

        self._base_url = f"http://127.0.0.1:{port}"
        self.log(f"\n✅ 启动中: http://{host}:{port}/v1")
        self.set_status("🟡 模型加载中...就绪后自动变绿并开始自检", "color:#b8860b; background:#fff8dc;")
        self._start_health_check()

    def _start_health_check(self):
        self.health_thread = HealthCheckThread(self._base_url, timeout_s=60)
        self.health_thread.ready.connect(self._on_server_ready)
        self.health_thread.warned_once.connect(lambda wth: self.log(f"🟠 {wth}"))
        self.health_thread.start()

    def _on_server_ready(self):
        self._ready_flag = True
        self.set_status("🟢 服务就绪", "color:#1a7f37; background:#dcf5e3;")
        port = self.ed_port.text().strip()
        self.ed_baseurl.setText(f"Base URL: http://127.0.0.1:{port}/v1    Model: {os.path.basename(self._current_model_path())}")
        self.log("✅ 服务就绪，开始端到端自检...")

    def _copy_baseurl(self):
        text = self.ed_baseurl.text()
        if text:
            # 只复制 Base URL 部分
            url = text.split("    Model:")[0].replace("Base URL: ", "").strip()
            QApplication.clipboard().setText(url)
            self.log(f"已复制 Base URL: {url}")
        self.selftest_thread = SelfTestThread(self._base_url)
        self.selftest_thread.result.connect(self._on_selftest)
        self.selftest_thread.start()

    def _on_selftest(self, ok, msg):
        if ok:
            self.log(f"✅ {msg}")
        else:
            self.log(f"⚠️ {msg}（服务本身已就绪，聊天仍可用，可手动在 Chat 测试）")

    def set_status(self, text, style):
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(f"padding:4px; font-size:13px; {style}")

    def on_stopped(self, host, port):
        ready = getattr(self, "_ready_flag", False)
        rc = getattr(getattr(self, "server_thread", None), "exit_rc", None)
        manual = getattr(self, "_manual_stop", False)
        self._ready_flag = False
        if hasattr(self, "txt_loadinfo") and self.txt_loadinfo:
            self.txt_loadinfo.clear()
        self._load_info = {}
        self._server_start = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.gb_gpu.setEnabled(True)
        if not ready and not manual and rc is not None and rc != 0:
            self.log(f"❌ llama-server 启动失败（退出码 {rc}），请查看上方日志——常见原因：显存不足、端口被占、后端 exe 与驱动不匹配")
            self.set_status("❌ 启动失败（退出码 %d），见日志" % rc, "color:#b41217; background:#fde8e8;")
            return
        self.log("服务已停止")
        self.set_status("⚪ 服务未启动", "")

    def stop_server(self):
        self._manual_stop = True
        for t in getattr(self, "health_thread", None), getattr(self, "selftest_thread", None):
            if t and t.isRunning():
                t.stop()
        if self.server_thread:
            self.server_thread.stop()

    def send_chat(self):
        text = self.ed_input.text().strip()
        if not text:
            return
        if not hasattr(self, "_base_url"):
            QMessageBox.warning(self, "提示", "请先启动服务")
            return
        if self.chat_thread and self.chat_thread.isRunning():
            return

        self.ed_input.clear()
        self.txt_chat.append(f'<p style="color:#4a9eff">你: {text}</p>')
        self.chat_history.append({"role": "user", "content": text})
        self.current_reply = ""
        self.txt_chat.append('<p style="color:#888">AI: </p>')

        self.chat_thread = ChatThread(self._base_url, self.chat_history, self.dsb_temp.value())
        self.chat_thread.token.connect(self._on_token)
        self.chat_thread.error.connect(lambda e: self.txt_chat.append(f"<span style='color:red'>错误: {e}</span>"))
        self.chat_thread.done.connect(self._on_done)
        self.chat_thread.start()

    def _on_token(self, tok):
        self.current_reply += tok
        cursor = self.txt_chat.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(tok)
        self.txt_chat.setTextCursor(cursor)

    def _on_done(self):
        self.txt_chat.append("")
        reply = self.current_reply.strip()
        if reply:
            self.chat_history.append({"role": "assistant", "content": reply})
        self.current_reply = ""

    def closeEvent(self, e):
        self.save_config()
        for t in [self.server_thread, self.chat_thread, self.health_thread,
                  self.selftest_thread, self.dl_thread]:
            if t and t.isRunning():
                t.stop()
                t.wait(3000)
        gt = getattr(self, "gpu_timer", None)
        if gt:
            gt.stop()
        e.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
