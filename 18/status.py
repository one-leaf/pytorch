import json, os, time, shutil, tempfile, sys
from contextlib import contextmanager
from datetime import datetime
from typing import Any
import numpy as np

if sys.platform != "win32":
    import fcntl

    def _lock_file(fd, exclusive=False, blocking=True):
        lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if not blocking:
            lock_type |= fcntl.LOCK_NB
        fcntl.flock(fd, lock_type)

    def _unlock_file(fd):
        fcntl.flock(fd, fcntl.LOCK_UN)

    @contextmanager
    def _shared_lock(fd):
        _lock_file(fd, exclusive=False)
        try:
            yield
        finally:
            _unlock_file(fd)

    @contextmanager
    def _exclusive_lock(fd):
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

model_name = ""
curr_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(curr_dir, 'model', model_name)
if not os.path.exists(model_dir): os.makedirs(model_dir)

# play.json: selfplay 写，train 读
# train.json: train 写，selfplay 不碰
play_file = os.path.join(model_dir, 'play.json')
play_file_bak = os.path.join(model_dir, 'play_bak.json')
train_file = os.path.join(model_dir, 'train.json')
train_file_bak = os.path.join(model_dir, 'train_bak.json')


# ── 公共工具 ──────────────────────────────────────────────────────────────────

def numpy_encoder(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def _atomic_write(filepath, state, model_dir_):
    """原子写入 + 排他锁（备份在调用方处理）"""
    fd = os.open(filepath, os.O_WRONLY | os.O_CREAT)
    try:
        with _exclusive_lock(fd):
            tmp_fd, tmp_path = tempfile.mkstemp(dir=model_dir_, suffix='.tmp')
            try:
                with os.fdopen(tmp_fd, 'w') as f:
                    json.dump(state, f, ensure_ascii=False, indent=4, default=numpy_encoder)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, filepath)
                os.chmod(filepath, 0o644)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
    finally:
        os.close(fd)


def _fill_defaults(state, defaults):
    """将 defaults 中缺失的 key 填入 state"""
    for section, section_defaults in defaults.items():
        if section not in state:
            state[section] = {}
        if isinstance(section_defaults, dict):
            for k, v in section_defaults.items():
                if k not in state[section]:
                    state[section][k] = v


# ── play.json: selfplay 专用 ──────────────────────────────────────────────────

def _default_play_state():
    return {
        "counters": {"agent": 0, "_agent": 0},
        "metrics": {
            # PPO player（selfplay，带 Dirichlet 噪声探索）
            "ppo_piececount": 0,
            "ppo_removedlines": 0,
            "ppo_steps": 0,
            "ppo_piececount_min": 9,
            "ppo_piececount_max": 0,
            "ppo_removedlines_best": 0,
            "ppo_piececount_best": 0,
            # test_play（纯贪婪，无噪声）
            "test_piececount": 0,
            "test_removedlines": 0,
            "test_steps": 0,
            "test_piececount_best": 0,
            "test_removedlines_best": 0,
        },
        "_accum": {"_sum_piececount": 0, "_sum_removedlines": 0, "_sum_steps": 0},
        "info": {},
    }


def save_play_state(state):
    """写入 play.json（selfplay 专用）"""
    format_str = '%Y-%m-%d %H:%M:%S'
    if "info" not in state:
        state["info"] = {}
    if "create" not in state["info"]:
        state["info"]["create"] = datetime.now().strftime(format_str)
    state["info"]["modify"] = datetime.now().strftime(format_str)

    # 备份
    try:
        if os.path.exists(play_file):
            shutil.copy(play_file, play_file_bak)
    except Exception:
        pass

    _atomic_write(play_file, state, model_dir)


def read_play_state():
    """读取 play.json"""
    state = _default_play_state()
    if not os.path.exists(play_file):
        return state

    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(play_file, 'r') as f:
                with _shared_lock(f):
                    state = json.load(f)
            break
        except (json.JSONDecodeError, ValueError):
            time.sleep(0.05)
            if attempt == max_retries - 1:
                raise
        except FileNotFoundError:
            time.sleep(0.05)
            if attempt == max_retries - 1:
                return _default_play_state()

    _fill_defaults(state, _default_play_state())
    return state


# ── train.json: train 专用 ────────────────────────────────────────────────────

def _default_train_state():
    return {
        "counters": {"train": 0, "_train": 0},
        "metrics": {
            "train_acc": 0,
            "train_kl": 0,
            "train_entropy": 0,
            "train_vloss": 0,
        },
        "training": {"kl": 1e-2, "lr_multiplier": 1, "entropy_weight": 1.0, "entropy_ema": 1.0},
        "history": [],
        "info": {},
    }


def _append_train_history(train_state, play_state):
    """每10轮训练(_train>=10)记录一次周期内快照，合并 play + train 指标"""
    tc = train_state.get("counters", {})
    tm = train_state.get("metrics", {})
    tr = train_state.get("training", {})
    pm = play_state.get("metrics", {})

    _train = tc.get("_train", 0)
    if _train < 10:
        return train_state

    snapshot = {
        "train": tc.get("train", 0),
        # PPO player
        "ppo_piececount": pm.get("ppo_piececount", 0),
        "ppo_removedlines": pm.get("ppo_removedlines", 0),
        "ppo_steps": pm.get("ppo_steps", 0),
        "ppo_piececount_min": pm.get("ppo_piececount_min", 999999),
        "ppo_piececount_max": pm.get("ppo_piececount_max", 0),
        "ppo_removedlines_best": pm.get("ppo_removedlines_best", 0),
        "ppo_piececount_best": pm.get("ppo_piececount_best", 0),
        # test_play
        "test_piececount": pm.get("test_piececount", 0),
        "test_removedlines": pm.get("test_removedlines", 0),
        "test_steps": pm.get("test_steps", 0),
        "test_piececount_best": pm.get("test_piececount_best", 0),
        "test_removedlines_best": pm.get("test_removedlines_best", 0),
        # train 训练指标
        "train_acc": tm.get("train_acc", 0),
        "train_kl": tm.get("train_kl", 0),
        "train_entropy": tm.get("train_entropy", 0),
        "train_vloss": tm.get("train_vloss", 0),
        # training
        "kl": tr.get("kl", 0),
        "lr_multiplier": tr.get("lr_multiplier", 1),
        "entropy_weight": tr.get("entropy_weight", 1.0),
        "entropy_ema": tr.get("entropy_ema", 1.0),
        "modify": "",
    }
    if "info" in train_state:
        snapshot["modify"] = train_state["info"].get("modify", "")

    train_state["history"].append(snapshot)
    tc["_train"] = 0
    return train_state


def save_train_state(state):
    """写入 train.json（train 专用）
    自动读取 play.json 最新状态，合并生成 history 快照"""
    format_str = '%Y-%m-%d %H:%M:%S'
    if "info" not in state:
        state["info"] = {}
    if "create" not in state["info"]:
        state["info"]["create"] = datetime.now().strftime(format_str)
    state["info"]["modify"] = datetime.now().strftime(format_str)

    # 读取 play.json 最新状态用于 history 快照
    play_state = read_play_state()
    _append_train_history(state, play_state)

    # 备份
    try:
        if os.path.exists(train_file):
            shutil.copy(train_file, train_file_bak)
    except Exception:
        pass

    # 归档：每 1000 轮 train 备份一次
    train_count = state["counters"].get("train", 0)
    if train_count > 0 and train_count % 1000 == 0:
        archive_path = os.path.join(model_dir, f'train_{train_count}.json')
        if not os.path.exists(archive_path) and os.path.exists(train_file):
            shutil.copy(train_file, archive_path)

    _atomic_write(train_file, state, model_dir)


def read_train_state():
    """读取 train.json"""
    state = _default_train_state()
    if not os.path.exists(train_file):
        return state

    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(train_file, 'r') as f:
                with _shared_lock(f):
                    state = json.load(f)
            break
        except (json.JSONDecodeError, ValueError):
            time.sleep(0.05)
            if attempt == max_retries - 1:
                raise
        except FileNotFoundError:
            time.sleep(0.05)
            if attempt == max_retries - 1:
                return _default_train_state()

    _fill_defaults(state, _default_train_state())
    return state


def set_train_value(state, key, value, rate=1/1000):
    """设置 training 下的值（支持滑动平均）"""
    if key not in state["training"]:
        state["training"][key] = value
    else:
        state["training"][key] += (value - state["training"][key]) * rate
