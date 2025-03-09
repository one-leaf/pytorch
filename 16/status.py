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
        result={"reward":[], "depth":[], "pacc":[], "vacc":[], "time":[], "piececount":[]}
    if "total" not in result:
        result["total"]={"agent":0, "pacc":0, "vacc":0, "ns":0, "depth":0, "step_time":0, "_agent":0}
    if "best" not in result:
        result["best"]={"reward":0, "agent":0}
    if "avg_piececount" not in result["total"]:
        result["total"]["avg_piececount"]=20                      
    if "min_piececount" not in result["total"]:
        result["total"]["min_piececount"]=20                      
    if "max_piececount" not in result["total"]:
        result["total"]["max_piececount"]=20                      
    if "n_playout" not in result["total"]:
        result["total"]["n_playout"]=64
    if "win_lost_tie" not in result["total"]:
        result["total"]["win_lost_tie"]=[0,0,0]            
    if "max_score" not in result["total"]:
        result["total"]["max_score"]=0  
    if "min_score" not in result["total"]:
        result["total"]["min_score"]=0  
    if "avg_score" not in result["total"]:
        result["total"]["avg_score"]=0  
    if "score_mcts" not in result["total"]:
        result["total"]["score_mcts"]=0  
    if "piececount_mcts" not in result["total"]:
        result["total"]["piececount_mcts"]=0
    if "piececount0_mcts" not in result["total"]:
        result["total"]["piececount0_mcts"]=0
    if "piececount1_mcts" not in result["total"]:
        result["total"]["piececount1_mcts"]=0
    if "qval" not in result["total"]:
        result["total"]["qval"]=0  
    if "max_qval" not in result["total"]:
        result["total"]["max_qval"]=0  
    if "min_qval" not in result["total"]:
        result["total"]["min_qval"]=0  
    if "state_value" not in result["total"]:
        result["total"]["state_value"]=0  
    if  "q_puct" not in result["total"]:
        result["total"]["q_puct"]=1
    if "piececount" not in result:
        result["piececount"]=[]
    if "piececount0_mcts" not in result:
        result["piececount0_mcts"]=[]
    if "piececount1_mcts" not in result:
        result["piececount1_mcts"]=[]
    if "update" not in result:
        result["update"]=[]
    if "qval" not in result:
        result["qval"]=[]  
    if "q_puct" not in result:
        result["q_puct"]=[]
    if "kl" not in result:
        result["kl"]=[]
    if "lr_multiplier" not in result["total"]:
        result["total"]["lr_multiplier"]=1
    if "optimizer_type" not in result["total"]:
        result["total"]["optimizer_type"]=0
    # if "advantage" in result: del result["advantage"]
    # if "avg_score_ex" in result["total"]: del result["total"]["avg_score_ex"]
    # if "exrewardRate" in result["total"]: del result["total"]["exrewardRate"]
    # if "avg_reward_piececount" in result["total"]: del result["total"]["avg_reward_piececount"]
    # if "avg_qval" in result["total"]: del result["total"]["avg_qval"]
    # if "piececount" in result["total"]: del result["total"]["piececount"]
    # if "avg_state_value" in result["total"]: del result["total"]["avg_state_value"]
    
    for key in result:
        if isinstance(result[key],list) and len(result[key])>30:            
            result[key]=result[key][-30:]
    
    return result

def set_status_total_value(result, key, value, rate=1/1000):
    if result["total"][key]==0:
        result["total"][key] = value
    else:
        result["total"][key] += (value-result["total"][key]) * rate
    