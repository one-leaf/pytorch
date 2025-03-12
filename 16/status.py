import pickle,json,os,time
from datetime import datetime
from filelock import FileLock

model_name = "vit-ti" # "vit" # "mlp"
curr_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(curr_dir, 'model', model_name)
if not os.path.exists(model_dir): os.makedirs(model_dir)
status_file = os.path.join(model_dir, 'status.json')
LOCK_FILE = status_file + '.lock'

def save_status_file(state):
    lock = FileLock(LOCK_FILE, timeout=10)
    with lock:    
        with open(status_file, 'w') as f:
            json.dump(state, f, ensure_ascii=False, indent=4)

def add_prop(state, key, default=0):
    if key not in state:
        state[key]=[]
    if key not in state["total"]:
        state["total"][key]=default

def add_total_prop(state, key, default=0):
    if key not in state["total"]:
        state["total"][key]=default

def read_status_file():
    # 获取历史训练数据
    state=None
    lock = FileLock(LOCK_FILE, timeout=10)
    with lock:    
        if os.path.exists(status_file):
            for i in range(5):
                try:
                    with open(status_file, "rb") as fn:
                        state = json.load(fn)
                    break
                except Exception as e:
                    time.sleep(10)                    
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
      
    add_total_prop(state, "min_piececount")    
    add_total_prop(state, "max_piececount")  
    add_total_prop(state, "n_playout", 64)
    add_total_prop(state, "win_lost_tie", [0,0,0])    
    add_total_prop(state, "max_score")
    add_total_prop(state, "min_score")
    
    add_prop(state, "score_mcts")
    add_prop(state, "piececount_mcts")
    add_prop(state, "q_avg")
    add_prop(state, "max_qval")
    add_prop(state, "min_qval")
    add_prop(state, "q_std")
    add_prop(state, "kl", 1e-2)
    add_total_prop(state, "lr_multiplier", 1)    
    add_total_prop(state, "optimizer_type", 0)    

    if "update" not in state:
        state["update"]=[]
    
    for key in state:
        if isinstance(state[key],list) and len(state[key])>30:            
            state[key]=state[key][-30:]
    
    return state

def set_status_total_value(state, key, value, rate=1/1000):
    if state["total"][key]==0:
        state["total"][key] = value
    else:
        state["total"][key] += (value-state["total"][key]) * rate
    