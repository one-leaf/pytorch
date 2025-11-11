import json,os,time,shutil
import  tempfile
from pathlib import Path
from contextlib import contextmanager
import errno
from datetime import datetime
import numpy as np

if os.name == 'posix':
    import fcntl
elif os.name == 'nt':
    pass

model_name = "vit-ti" # "vit" # "mlp"
curr_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(curr_dir, 'model', model_name)
if not os.path.exists(model_dir): os.makedirs(model_dir)
status_file = os.path.join(model_dir, 'status.json')
status_file_bak = os.path.join(model_dir, 'status_bak.json')
status_lock_file = os.path.join(model_dir, 'status.lock')

@contextmanager
def file_lock(lock_path):
    """文件锁上下文管理器"""
    lock_file = Path(lock_path)
    
    # 等待锁释放
    while lock_file.exists():
        time.sleep(0.1)
    
    # 创建锁文件
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # 如果其他进程先创建了，重新等待
            while lock_file.exists():
                time.sleep(0.1)
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        else:
            raise
    
    try:
        yield lock_file
    finally:
        # 确保清理锁文件
        try:
            if lock_file.exists():
                lock_file.unlink()
        except:
            pass

def numpy_encoder(obj):
    """
    自定义 JSON 编码函数，用于处理 NumPy 类型。
    """
    # 检查是否是 NumPy 整数类型
    if isinstance(obj, np.integer):
        return int(obj)
    # 检查是否是 NumPy 浮点数类型 (如 float32, float64)
    elif isinstance(obj, np.floating):
        # 使用 .item() 方法是提取 NumPy 标量到 Python 标量的推荐方式
        return float(obj) 
    # 检查是否是 NumPy 数组
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # 如果对象类型不是我们定义的，则让 JSON 编码器继续抛出 TypeError
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def save_status_file(state):  
    format_str = '%Y-%m-%d %H:%M:%S'
    if "info" not in state:
        state["info"] = {"create":datetime.now().strftime(format_str)}

    state["info"]["modify"] = datetime.now().strftime(format_str)  
    
    start_dt = datetime.strptime(state["info"]["create"], format_str)
    end_dt   = datetime.strptime(state["info"]["modify"], format_str)
    delta = end_dt - start_dt
    if delta.days>0:
        file_path = os.path.join(model_dir, f'status_{delta.days}d.json')
        if not os.path.exists(file_path):
            shutil.copy(status_file, file_path)
    if state["total"]["agent"]%1000 == 0:
        file_path = os.path.join(model_dir, f'status_{state["total"]["agent"]}.json')
        if not os.path.exists(file_path) and os.path.exists(status_file):
            shutil.copy(status_file, file_path)
    
    with file_lock(status_lock_file):
        temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', newline='',dir=model_dir)
        with temp_file as f:
            # lock_file(f, exclusive=True)
            json.dump(state, f, ensure_ascii=False, indent=4, default=numpy_encoder)
            # unlock_file(f)
        if os.path.exists(status_file):
            os.remove(status_file)
        shutil.move(temp_file.name, status_file)
        os.chmod(status_file, 0o644)

def add_prop(state, key, default=0):
    if key not in state:
        state[key]=[]
    if key not in state["total"]:
        state["total"][key]=default

def add_total_prop(state, key, default=0):
    if key not in state["total"]:
        state["total"][key]=default

def read_status_file(max_keep=30):
    # 获取历史训练数据
    state=None
    while os.path.exists(status_lock_file): 
        time.sleep(0.1)

    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                state = json.load(f)
            shutil.copy(status_file, status_file_bak)    
        except Exception as e:
            os.remove(status_file)
            if os.path.exists(status_file_bak):
                shutil.move(status_file_bak, status_file)
            raise e        
                      
    if state==None:
        state={"total":{"agent":0, "_agent":0}}
    if "best" not in state:
        state["best"]={"score":0, "agent":0}
        
    add_prop(state, "score")
    add_prop(state, "depth")
    add_prop(state, "pacc")
    add_prop(state, "step_time")
    add_prop(state, "piececount")    
    add_prop(state, "steps")    
      
    add_total_prop(state, "min_piececount")    
    add_total_prop(state, "max_piececount")  
    add_total_prop(state, "n_playout", 64)
    add_total_prop(state, "p_n_q", [0,0,0])    
    add_total_prop(state, "model_probs", [0,0,0,0,0])
    add_total_prop(state, "mcts_probs", [0,0,0,0,0])
    add_total_prop(state, "max_score")
    
    add_prop(state, "no_terminal_rate", 0)
    add_prop(state, "min_score")
    
    add_prop(state, "score_mcts")
    add_prop(state, "piececount_mcts")
    add_prop(state, "q_avg")
    add_prop(state, "q_std")
    add_prop(state, "p_std", 1)
    add_total_prop(state, "kl", 1e-2)
    add_total_prop(state, "lr_multiplier", 1)    
    add_prop(state, "steps_mcts", 0)    
    add_prop(state, "find_end_steps", 50)    
    

    if "update" not in state:
        state["update"]=[]
    
    for key in state:
        if isinstance(state[key],list) and len(state[key])>max_keep: 
            max_v = max(state[key])
            min_v = min(state[key])
            need_keep = key not in ["update"] and max_v==state[key][0]
            while len(state[key])>max_keep:
                state[key].pop(0)  
            if need_keep:
                state[key][0] = max_v
                state[key][1] = min_v 
    return state

def set_status_total_value(state, key, value, rate=1/1000):
    if type(state["total"][key]) == list:
        for i in range(len(value)):
            state["total"][key][i] = round(state["total"][key][i] + (value[i]-state["total"][key][i]) * rate, 3)
    else:    
        state["total"][key] += (value-state["total"][key]) * rate
    