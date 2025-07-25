import json,os,time,shutil
import  tempfile

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

def lock_file(f, exclusive=True):
    """
    给文件对象 f 加锁。
    - POSIX 系统使用 fcntl.flock。
    - Windows 系统使用 msvcrt.locking。
    """
    if os.name == 'posix':
        lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(f.fileno(), lock_type)
    elif os.name == 'nt':
        pass

def unlock_file(f):
    """
    解锁文件对象 f。
    """
    if os.name == 'posix':
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    elif os.name == 'nt':
        pass

def save_status_file(state):   
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', newline='')
    with temp_file as f:
        # lock_file(f, exclusive=True)
        json.dump(state, f, ensure_ascii=False, indent=4)
        # unlock_file(f)
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
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                state = json.load(f)
            shutil.copy(status_file, status_file_bak)    
        except Exception as e:
            shutil.copy(status_file_bak, status_file)
            raise e                  
    if state==None:
        state={"total":{"agent":0, "_agent":0}}
    if "best" not in state:
        state["best"]={"score":0, "agent":0}
        
    add_prop(state, "score")
    add_prop(state, "depth")
    add_prop(state, "pacc")
    add_prop(state, "vdiff")
    add_prop(state, "step_time")
    add_prop(state, "piececount")    
    add_prop(state, "steps")    
      
    add_total_prop(state, "min_piececount")    
    add_total_prop(state, "max_piececount")  
    add_total_prop(state, "n_playout", 64)
    add_total_prop(state, "win_lost_tie", [0,0,0])    
    add_total_prop(state, "max_score")
    
    add_prop(state, "no_terminal_rate", 0)
    add_prop(state, "min_score")
    
    add_prop(state, "score_mcts")
    add_prop(state, "piececount_mcts")
    add_prop(state, "q_avg")
    add_prop(state, "q_std")
    add_prop(state, "kl", 1e-2)
    add_total_prop(state, "lr_multiplier", 1)    
    add_total_prop(state, "optimizer_type", 0)    
    add_prop(state, "steps_mcts", 0)    
    add_prop(state, "rate_mcts", 0)    
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
    