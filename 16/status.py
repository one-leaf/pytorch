import json,os,time,shutil
import  tempfile
from pathlib import Path
from contextlib import contextmanager
import errno
from datetime import datetime

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

def save_status_file(state):  
    if "info" not in state:
        state["info"] = {"create":datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    state["info"]["modify"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  
    
    with file_lock(status_lock_file):
        temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', newline='',dir=model_dir)
        with temp_file as f:
            # lock_file(f, exclusive=True)
            json.dump(state, f, ensure_ascii=False, indent=4)
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
    add_total_prop(state, "optimizer_type", 0)    
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
    if state["total"][key]==0:
        state["total"][key] = value
    else:
        state["total"][key] += (value-state["total"][key]) * rate
    