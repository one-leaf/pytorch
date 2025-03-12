import pickle,json,os,time
from datetime import datetime

model_name = "vit-ti" # "vit" # "mlp"
curr_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(curr_dir, 'model', model_name)
if not os.path.exists(model_dir): os.makedirs(model_dir)
status_file = os.path.join(model_dir, 'status.json')

def save_status_file(result):
    with open(status_file+"_pkl", 'wb') as fn:
        pickle.dump(result, fn)
    with open(status_file, 'w') as f:
        json.dump(result, f, ensure_ascii=False)

def add_prop(result, key, default=0):
    if key not in result:
        result[key]=[]
    if key not in result["total"]:
        result["total"][key]=default

def add_total_prop(result, key, default=0):
    if key not in result["total"]:
        result["total"][key]=default

def read_status_file():
    # 获取历史训练数据
    result=None
    if os.path.exists(status_file):
        for i in range(5):
            try:
                with open(status_file, "rb") as fn:
                    result = json.load(fn)
                break
            except Exception as e:
                time.sleep(10)
            try:
                with open(status_file+"_pkl", "rb") as fn:
                    result = pickle.load(fn)
                break
            except Exception as e:
                time.sleep(10)

        if result==None:
            ext = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            os.replace(status_file, status_file+"_"+ext) 
    if result==None:
        result={"total":{"agent":0, "_agent":0}}
    if "best" not in result:
        result["best"]={"score":0, "agent":0}
        
    add_prop(result, "score")
    add_prop(result, "depth")
    add_prop(result, "pacc")
    add_prop(result, "vdiff")
    add_prop(result, "step_time")
    add_prop(result, "ns")
    add_prop(result, "piececount")    
    add_prop(result, "avg_piececount")    
    add_prop(result, "min_piececount")    
    add_prop(result, "max_piececount")  
      
    add_total_prop(result, "n_playout", 64)
    add_total_prop(result, "win_lost_tie", [0,0,0])    
        
    add_prop(result, "max_score")
    add_prop(result, "min_score")
    add_prop(result, "avg_score")
    add_prop(result, "score_mcts")
    add_prop(result, "piececount_mcts")
    add_prop(result, "q_avg")
    add_prop(result, "max_qval")
    add_prop(result, "min_qval")
    add_prop(result, "q_std")
    add_prop(result, "kl", 1e-2)
    add_total_prop(result, "lr_multiplier", 1)    
    add_total_prop(result, "optimizer_type", 0)    

    if "update" not in result:
        result["update"]=[]
    
    for key in result:
        if isinstance(result[key],list) and len(result[key])>30:            
            result[key]=result[key][-30:]
    
    return result

def set_status_total_value(result, key, value, rate=1/1000):
    if result["total"][key]==0:
        result["total"][key] = value
    else:
        result["total"][key] += (value-result["total"][key]) * rate
    