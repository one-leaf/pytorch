import json,os,time,shutil,tempfile,sys
from contextlib import contextmanager
from datetime import datetime
from typing import Any
import numpy as np

if sys.platform != "win32":
    import fcntl

    def _lock_file(fd, exclusive=False, blocking=True):
        """内核级文件锁（Docker 挂载卷安全）"""
        lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if not blocking:
            lock_type |= fcntl.LOCK_NB
        fcntl.flock(fd, lock_type)

    def _unlock_file(fd):
        fcntl.flock(fd, fcntl.LOCK_UN)

    @contextmanager
    def _shared_lock(fd):
        """共享读锁上下文"""
        _lock_file(fd, exclusive=False)
        try:
            yield
        finally:
            _unlock_file(fd)

    @contextmanager
    def _exclusive_lock(fd):
        """排他写锁上下文"""
        _lock_file(fd, exclusive=True)
        try:
            yield
        finally:
            _unlock_file(fd)
else:
    def _lock_file(fd, exclusive=False, blocking=True):
        pass

    def _unlock_file(fd):
        pass

    @contextmanager
    def _shared_lock(fd):
        yield

    @contextmanager
    def _exclusive_lock(fd):
        yield

model_name = "vit-ti" # "vit" # "mlp"
curr_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(curr_dir, 'model', model_name)
if not os.path.exists(model_dir): os.makedirs(model_dir)
status_file = os.path.join(model_dir, 'status.json')
status_file_bak = os.path.join(model_dir, 'status_bak.json')
HISTORY_MAX = 100


def _default_state():
    return {
        "counters": {"agent": 0, "_agent": 0},
        "metrics": {
            "grpo_piececount": 0,
            "grpo_removedlines": 0, "grpo_steps": 0,
            "grpo_piececount_min": 999999, "grpo_piececount_max": 0,
            "grpo_removedlines_min": 999999, "grpo_removedlines_max": 0,
            "grpo_removedlines_best": 0, "grpo_piececount_best": 0,
            "grpo_removedlines_worst": 999999, "grpo_piececount_worst": 999999,
        },
        "training": {"kl": 1e-2, "lr_multiplier": 1},
        "_accum": {"_sum_piececount": 0, "_sum_removedlines": 0, "_sum_steps": 0},
        "history": [],
        "info": {},
    }


def _migrate(state: dict[str, Any]):
    """兼容旧格式：将 state["total"] 扁平结构迁移到新的分层结构"""
    if "_migrated" in state:
        return
    if "total" not in state or "counters" in state:
        state.setdefault("_migrated", True)
        return

    old = state.pop("total")
    state.setdefault("counters", {})
    state.setdefault("metrics", {})
    state.setdefault("training", {})
    state.setdefault("_accum", {})
    state.setdefault("history", [])

    for k in ("agent", "_agent"):
        if k in old:
            state["counters"][k] = old.pop(k)
    for k in ["grpo_piececount", "grpo_removedlines", "grpo_steps",
              "grpo_piececount_min", "grpo_piececount_max",
              "grpo_removedlines_min", "grpo_removedlines_max",
              "grpo_removedlines_best", "grpo_piececount_best",
              "grpo_removedlines_worst", "grpo_piececount_worst"]:
        if k in old:
            state["metrics"][k] = old.pop(k)
    for k in ("kl", "lr_multiplier"):
        if k in old:
            state["training"][k] = old.pop(k)
    for k in ("_sum_piececount", "_sum_removedlines", "_sum_steps"):
        if k in old:
            state["_accum"][k] = old.pop(k)
    if "history" in old:
        state["history"] = old.pop("history")

    for k in ['score', 'depth', 'pacc', 'step_time', 'piececount', 'steps',
              'min_piececount', 'max_piececount', 'max_score', 'no_terminal_rate',
              'min_score', 'find_end_steps', 'update', 'best']:
        state.pop(k, None)

    state.setdefault("_migrated", True)


def numpy_encoder(obj):
    """自定义 JSON 编码函数，用于处理 NumPy 类型。"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def _append_history(state: dict[str, Any]):
    """每100轮(_agent>100)记录一次周期内平均值快照"""
    c = state.get("counters", {})
    m = state.get("metrics", {})
    tr = state.get("training", {})
    acc = state.get("_accum", {})

    _agent = c.get("_agent", 0)
    if _agent < 100:
        return

    n = max(_agent, 1)
    snapshot = {
        "agent": c.get("agent", 0),
        "grpo_piececount": round(acc.get("_sum_piececount", 0) / n, 3),
        "grpo_removedlines": round(acc.get("_sum_removedlines", 0) / n, 3),
        "grpo_steps": round(acc.get("_sum_steps", 0) / n, 3),
        "grpo_piececount_min": m.get("grpo_piececount_min", 999999),
        "grpo_piececount_max": m.get("grpo_piececount_max", 0),
        "grpo_removedlines_min": m.get("grpo_removedlines_min", 999999),
        "grpo_removedlines_max": m.get("grpo_removedlines_max", 0),
        "grpo_removedlines_best": m.get("grpo_removedlines_best", 0),
        "grpo_piececount_best": m.get("grpo_piececount_best", 0),
        "grpo_removedlines_worst": m.get("grpo_removedlines_worst", 999999),
        "grpo_piececount_worst": m.get("grpo_piececount_worst", 999999),
        "kl": tr.get("kl", 0),
        "lr_multiplier": tr.get("lr_multiplier", 1),
        "modify": "",
    }
    if "info" in state:
        snapshot["modify"] = state["info"].get("modify", "")

    state["history"].append(snapshot)
    if len(state["history"]) > HISTORY_MAX:
        state["history"] = state["history"][-HISTORY_MAX:]

    c["_agent"] = 0
    acc["_sum_piececount"] = 0
    acc["_sum_removedlines"] = 0
    acc["_sum_steps"] = 0


def save_status_file(state:dict[str, Any]):
    format_str = '%Y-%m-%d %H:%M:%S'
    if "info" not in state:
        state["info"] = {}
    if "create" not in state["info"]:
        state["info"]["create"] = datetime.now().strftime(format_str)
    state["info"]["modify"] = datetime.now().strftime(format_str)

    _migrate(state)
    _append_history(state)

    # 归档
    start_dt = datetime.strptime(state["info"]["create"], format_str)
    end_dt   = datetime.strptime(state["info"]["modify"], format_str)
    delta = end_dt - start_dt
    if delta.days > 0:
        file_path = os.path.join(model_dir, f'status_{delta.days}d.json')
        if not os.path.exists(file_path):
            shutil.copy(status_file, file_path)
    if state["counters"]["agent"] % 1000 == 0:
        file_path = os.path.join(model_dir, f'status_{state["counters"]["agent"]}.json')
        if not os.path.exists(file_path) and os.path.exists(status_file):
            shutil.copy(status_file, file_path)

    # 原子写入 + 排他锁
    fd = os.open(status_file, os.O_WRONLY | os.O_CREAT)
    try:
        with _exclusive_lock(fd):
            # 先写临时文件，再排他锁原文件后原子替换
            tmp_fd, tmp_path = tempfile.mkstemp(dir=model_dir, suffix='.tmp')
            try:
                with os.fdopen(tmp_fd, 'w') as f:
                    json.dump(state, f, ensure_ascii=False, indent=4, default=numpy_encoder)
                    f.flush()
                    os.fsync(f.fileno())

                os.replace(tmp_path, status_file)
                os.chmod(status_file, 0o644)
            except:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
    finally:
        os.close(fd)


def read_status_file():
    state = _default_state()

    if not os.path.exists(status_file):
        return state

    # 共享读锁：不阻塞写者，也不被写者阻塞（Linux 上 flock 是 advisory）
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(status_file, 'r') as f:
                with _shared_lock(f):
                    state = json.load(f)
                    shutil.copy(status_file, status_file_bak)
            break
        except (json.JSONDecodeError, ValueError):
            # 写操作进行中，短暂等待后重试
            time.sleep(0.05)
            if attempt == max_retries - 1:
                raise
        except FileNotFoundError:
            time.sleep(0.05)
            if attempt == max_retries - 1:
                return _default_state()

    _migrate(state)
    for section, defaults in _default_state().items():
        if section not in state:
            state[section] = {}
        if isinstance(defaults, dict):
            for k, v in defaults.items():
                if k not in state[section]:
                    state[section][k] = v

    return state


def set_status_value(state:dict[str, Any], key:str, value:Any, rate=1/1000):
    """设置 training 下的值（支持滑动平均）"""
    if key not in state["training"]:
        state["training"][key] = value
    else:
        state["training"][key] += (value - state["training"][key]) * rate
