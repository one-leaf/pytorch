import os, glob, pickle

from model import PolicyValueNet, data_dir, data_wait_dir, model_file
from agent_numba import Agent, ACTIONS
from mcts_single_numba import MCTSPlayer

import time, json
from datetime import datetime, timedelta

from itertools import count
import os, random, uuid, math

import numpy as np
import copy

# 定义游戏的动作
GAME_ACTIONS_NUM = len(ACTIONS) 
GAME_WIDTH, GAME_HEIGHT = 10, 20

class Train():
    def __init__(self):
        self.game_batch_num = 1000000  # selfplay对战次数
        self.batch_size = 512     # data_buffer中对战次数超过n次后开始启动模型训练

        # training params
        self.learn_rate = 1e-4
        self.lr_multiplier = 1.0  # 基于KL的自适应学习率
        self.temp = 1  # MCTS的概率参数，越大越不肯定，训练时1，预测时1e-3
        self.n_playout = 256  # 每个动作的模拟战记录个数，影响后续 512/2 = 256；256/16 = 16个方块 的走法
        # 64/128/256/512 都不行
        # step -> score
        # 128  --> 0.7
        # self.n_playout = 128  # 每个动作的模拟战记录个数，影响后续 128/2 = 66；64/16 = 4个方块 的走法
        self.play_size = 10 # 每次测试次数
        self.buffer_size = 1000000  # cache对次数
        self.epochs = 2  # 每次更新策略价值网络的训练步骤数, 推荐是5
        self.kl_targ = 0.02  # 策略价值网络KL值目标
        self.best_win_ratio = 0.0
        
        # aplhazero 的最佳值是 4 
        # aplhatensor 是 5
        # MCTS child权重， 用来调节MCTS搜索深度，越大搜索越深，越相信概率，越小越相信Q 的程度 默认 5
        # 由于value完全用结果胜负来拟合，所以value不稳，只能靠概率p拟合，最后带动value来拟合
        self.c_puct = 10  
        self.q_puct = 1
        self.max_step_count = 10000 
        self.limit_steptime = 3  # 限制每一步的平均花费时间，单位秒，默认10秒

        # 等待训练的序列
        self.waitplaydir=os.path.join(data_dir,"play")
        if not os.path.exists(self.waitplaydir): os.makedirs(self.waitplaydir)


    def save_status_file(self, result, status_file):
        with open(status_file+"_pkl", 'wb') as fn:
            pickle.dump(result, fn)
        with open(status_file, 'w') as f:
            json.dump(result, f, ensure_ascii=False)


    def read_status_file(self, status_file):
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
            result["total"]["n_playout"]=self.n_playout
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
        # if "advantage" in result: del result["advantage"]
        # if "avg_score_ex" in result["total"]: del result["total"]["avg_score_ex"]
        # if "exrewardRate" in result["total"]: del result["total"]["exrewardRate"]
        # if "avg_reward_piececount" in result["total"]: del result["total"]["avg_reward_piececount"]
        # if "avg_qval" in result["total"]: del result["total"]["avg_qval"]
        # if "piececount" in result["total"]: del result["total"]["piececount"]
        # if "avg_state_value" in result["total"]: del result["total"]["avg_state_value"]
        
        return result

    def get_equi_data(self, states, mcts_probs, values, scores):
        """
        通过翻转增加数据集
        play_data: [(state, mcts_prob, values, score), ..., ...]
        """
        extend_data = []
        for i in range(len(states)):
            state, mcts_prob, value, score=states[i], mcts_probs[i], values[i], scores[i]
            extend_data.append((state, mcts_prob, value, score))
            if mcts_prob[0]<0.2 and np.max(mcts_prob)>0.8: # 如果旋转的概率的不大，就做翻转
                equi_state = np.array([np.fliplr(s) for s in state])
                equi_mcts_prob = mcts_prob[[0,2,1,3,4]]
                extend_data.append((equi_state, equi_mcts_prob, value, score))
            # if i==0:
            #     print("state:",state)
            #     print("mcts_prob:",mcts_prob)
            #     print("equi_state:",equi_state)
            #     print("equi_mcts_prob:",equi_mcts_prob)
            #     print("value:",value)
        return extend_data

    def test_play(self,game_json,policy_value_net):
        # 先运行测试
        min_his_pieces = None
        min_his_pieces_len = 0
        min_pieces_count = -1
        min_removedlines = -1
        max_pieces_count = -1
        max_removedlines = -1
        for _ in range(self.play_size):
            result = self.read_status_file(game_json)     
            agent = Agent(isRandomNextPiece=True)
            start_time = time.time()
            agent.show_mcts_process= False
            # agent.id = 0

            for i in range(self.max_step_count):
                action = policy_value_net.policy_value_fn_best_act(agent)
                _, reward = agent.step(action)
                if reward > 0:
                    print("#"*40, 'score:', agent.removedlines, 'height:', agent.pieceheight, 'piece:', agent.piececount, "shape:", agent.fallpiece["shape"], \
                        'step:', agent.steps, "step time:", round((time.time()-start_time)/i,3),'avg_score:', result["total"]["avg_score"])            

                if agent.terminal:            
                    result["total"]["avg_score"] += (agent.removedlines-result["total"]["avg_score"])/1000
                    result["total"]["avg_piececount"] += (agent.piececount-result["total"]["avg_piececount"])/1000
                    result["lastupdate"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    break
                
            self.save_status_file(result, game_json)

            agent.print()

            # 找到当前放置的最小方块局面，重新玩
            if min_pieces_count==-1 or agent.piececount < min_pieces_count:
                min_pieces_count = agent.piececount
                min_his_pieces = agent.piecehis
                min_his_pieces_len = len(agent.piecehis)
                min_removedlines = agent.removedlines
            if max_pieces_count==-1 or agent.piececount > max_pieces_count:         
                max_pieces_count = agent.piececount
                max_removedlines = agent.removedlines

            if agent.piececount<result["total"]["min_piececount"]:
                filename = "{}-{}.pkl".format("".join(min_his_pieces), min_his_pieces_len)
                his_pieces_file = os.path.join(self.waitplaydir, filename)
                print("save need replay", his_pieces_file)
                with open(his_pieces_file, "wb") as fn:
                    pickle.dump(min_his_pieces, fn)
        
        result["total"]["max_score"] += (max_removedlines-result["total"]["max_score"])/100
        result["total"]["max_piececount"] += (max_pieces_count-result["total"]["max_piececount"])/100
        result["total"]["min_score"] += (min_removedlines-result["total"]["min_score"])/100
        result["total"]["min_piececount"] += (min_pieces_count-result["total"]["min_piececount"])/100                
        self.save_status_file(result, game_json)          
                               
        return min_removedlines, min_his_pieces, min_his_pieces_len

    def play(self, cache, result, min_removedlines, his_pieces, his_pieces_len, player):
        data = {"steps":[],"shapes":[],"last_state":0,"score":0,"piece_count":0}
        if his_pieces!=None:
            print("min_removedlines:", min_removedlines, "pieces_count:", len(his_pieces))
            print("his_pieces:", his_pieces)
            agent = Agent(isRandomNextPiece=False, nextPiecesList=his_pieces)
            agent.is_replay = True
            agent.limitstep = True
        else:
            # 新局按Q值走，探索
            agent = Agent(isRandomNextPiece=False, )
            agent.is_replay = False
            agent.limitstep = False
            
        agent.exreward = True
        agent.setCache(cache)
        
        agent.show_mcts_process= True
        
        max_emptyCount = random.randint(10,30)
        start_time = time.time()
        mark_reward_piececount = -1
        total_qval=0
        max_qval=0
        min_qval=0
        total_state_value=0
        need_max_ps = False # random.random()>0.5
        exreward_end_piececount = 0
        print("exreward:", agent.exreward,"max_emptyCount:",max_emptyCount,"isRandomNextPiece:",agent.isRandomNextPiece,"limitstep:",agent.limitstep,"max_ps:",need_max_ps,"max_qs:",agent.is_replay)
        for i in range(self.max_step_count):
            _step={"step":i}
            _step["state"] = np.copy(agent.current_state())
            # print(_step["state"][0])           
            _step["piece_count"] = agent.piececount               
            _step["shape"] = agent.fallpiece["shape"]
            _step["pre_piece_height"] = agent.pieceheight

            # if agent.piecesteps==0 and random.random()>0.5: 
            #     need_max_ps=not need_max_ps
            #     print("switch need_max_ps to:", need_max_ps)
            # need_max_ps = random.random() < agent.removedlines/100   
            action, qval, move_probs, state_value, acc_ps, depth, ns = player.get_action(agent, temp=1) 

            _, reward = agent.step(action)

            total_qval += qval
            if agent.exreward:            
                max_qval = max(max_qval, qval)
                min_qval = min(min_qval, qval)    
            # if qval > 0:
            #     avg_qval += 1
            # else:
            #     avg_qval += -1

            total_state_value += state_value 

            _step["piece_height"] = agent.pieceheight
            _step["reward"] = reward if reward>0 else 0
            _step["move_probs"] = move_probs
            _step["state_value"] = state_value
            _step["qval"] = qval
            _step["acc_ps"] = acc_ps
            _step["depth"] = depth
            _step["ns"] = ns
            _step["score"] = agent.score

            data["steps"].append(_step)

            # time.sleep(0.1)

            # 这里的奖励是消除的行数
            if agent.state==1:
                if agent.piececount-agent.last_reward==1:
                    repeat_count = 40
                    print("#"*repeat_count, 'score:', agent.score, "reward:",reward, 'qval', round(qval,2), 'height:', agent.pieceheight, 'piece:', agent.piececount, \
                        'step:', agent.steps, "step time:", round((time.time()-start_time)/i,3),'reward_p:', agent.piececount-mark_reward_piececount)
                agent.print()
                if agent.piececount%2==0 and (player.need_max_ps or player.need_max_ns):
                    player.need_max_ps = not player.need_max_ps
                    player.need_max_ns = not player.need_max_ns
                
                if agent.exreward: exreward_end_piececount = agent.piececount
                    
                if not os.path.exists(self.waitplaydir):
                    raise Exception("waitplaydir not exists")

            # 如果游戏结束或玩了超过1小时或10个方块都没有消除一行
            paytime = time.time()-start_time
                # (agent.removedlines > result["total"]["avg_score"]+1)  or \
            if agent.terminal:# or (agent.state==1 and paytime>60*60):# or agent.piececount-agent.last_reward>=8:
                data["score"] = agent.score
                data["piece_count"] = agent.piececount
                data["piece_height"] = agent.pieceheight
                return agent, data, total_qval, total_state_value, exreward_end_piececount, max_qval, min_qval, start_time, paytime
            


    def collect_selfplay_data(self):
        """收集自我对抗数据用于训练"""       
        print("TRAIN Self Play starting ...")
        
        # 这里用上一次训练的模型作为预测有助于模型稳定
        load_model_file=model_file
        if os.path.exists(model_file+".bak"):
            load_model_file = model_file+".bak"

        if os.path.exists(load_model_file):
            if time.time()-os.path.getmtime(load_model_file)>120*60:
                print("超过120分钟模型都没有更新了，停止训练")
                time.sleep(60)
                return

        # 开始游戏
        policy_value_net = PolicyValueNet(GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=load_model_file)
        bestmodelfile = model_file+"_best"
        
        game_json = os.path.join(data_dir, "result.json")
  
        print('start test time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        min_removedlines, his_pieces, his_pieces_len = self.test_play(game_json, policy_value_net)
        
        print('end test time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # 正式运行
        limit_depth=20
        result = self.read_status_file(game_json) 
        
        if result["total"]["depth"]>limit_depth:
            limit_depth=result["total"]["depth"]
        
        self.n_playout = int(result["total"]["n_playout"])

        self.q_puct = result["total"]["q_puct"]
        player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=self.c_puct, q_puct=self.q_puct, n_playout=self.n_playout, limit_depth=limit_depth)

        cache={}

        # 检查有没有需要重复运行的
        listFiles = [f for f in os.listdir(self.waitplaydir) if f.endswith(".pkl")]
        if listFiles:
            earliest_file = min(listFiles, key=lambda f: os.path.getctime(os.path.join(self.waitplaydir, f)))
            filename = os.path.join(self.waitplaydir, earliest_file)
            with open(filename, "rb") as fn:
                his_pieces = pickle.load(fn)
                his_pieces_len = len(his_pieces)
            print("load need replay", filename)
            os.remove(filename)
        else:
            his_pieces = []
            his_pieces_len = 0
        
        play_data = []
        result = self.read_status_file(game_json) 
        exreward_end_piececounts = []    
        for playcount in range(2):
            player.set_player_id(playcount)
            if playcount==0:
                player.need_max_ps = False
                player.need_max_ns = True
            elif playcount==1:
                player.need_max_ps = True
                player.need_max_ns = False
                
            agent, data, qval, state_value, exreward_end_piececount, max_qval, min_qval, start_time, paytime = self.play(cache, result, min_removedlines, his_pieces, his_pieces_len, player)
            play_data.append({"agent":agent, "data":data, "qval":qval, "max_qval":max_qval, "min_qval":min_qval, "state_value":state_value, "start_time":start_time, "paytime":paytime})
            his_pieces = agent.piecehis
            his_pieces_len = len(agent.piecehis)
            exreward_end_piececounts.append(exreward_end_piececount)
                
        print("TRAIN Self Play ending ...")
                
        total_game_score =  play_data[0]["agent"].removedlines + play_data[1]["agent"].removedlines 
        total_game_steps =  play_data[0]["agent"].steps + play_data[1]["agent"].steps 
        total_game_piececount =  play_data[0]["agent"].piececount + play_data[1]["agent"].piececount 
        total_game_exreward_end_piececounts =  exreward_end_piececounts[0] + exreward_end_piececounts[1] 
        total_game_paytime =  play_data[0]["paytime"] + play_data[1]["paytime"] 
        total_game_state_value =  play_data[0]["state_value"] + play_data[1]["state_value"] 
        total_game_qval =  play_data[0]["qval"] + play_data[1]["qval"] 
        max_game_qval = max(play_data[0]["max_qval"], play_data[1]["max_qval"])
        min_game_qval = min(play_data[0]["min_qval"], play_data[1]["min_qval"])
        game_score = max(play_data[0]["agent"].removedlines, play_data[1]["agent"].removedlines)
        win_values =[-1, -1]
        if play_data[0]["agent"].piececount>play_data[1]["agent"].piececount:
            win_values[0] = 1
        elif play_data[0]["agent"].piececount<play_data[1]["agent"].piececount:
            win_values[1] = 1
        elif exreward_end_piececounts[0]>exreward_end_piececounts[1]:
            win_values[0] = 1
        elif exreward_end_piececounts[0]<exreward_end_piececounts[1]:
            win_values[1] = 1

        result = self.read_status_file(game_json)
        
        steptime = total_game_paytime/total_game_steps            
        avg_qval = total_game_qval/total_game_steps
        avg_state_value = total_game_state_value/total_game_steps
        
        print("step pay time:", steptime, "qval:", avg_qval, "avg_state_value:", avg_state_value)
                        
        alpha = 0.01
        result["total"]["score_mcts"] += alpha * (total_game_score/2-result["total"]["score_mcts"])
        result["total"]["piececount_mcts"] += alpha * (total_game_piececount/2-result["total"]["piececount_mcts"])   
        result["total"]["piececount0_mcts"] += alpha * (total_game_exreward_end_piececounts/2-result["total"]["piececount0_mcts"])
        result["total"]["piececount1_mcts"] += alpha * ((total_game_piececount-total_game_exreward_end_piececounts)/2 - result["total"]["piececount1_mcts"])    
        result["total"]["qval"] += alpha * (avg_qval - result["total"]["qval"])            
        result["total"]["max_qval"] += alpha * (max_game_qval - result["total"]["max_qval"])            
        result["total"]["min_qval"] += alpha * (min_game_qval - result["total"]["min_qval"])
        result["total"]["state_value"] += alpha * (avg_state_value - result["total"]["state_value"])
        result["total"]["step_time"] += alpha * (steptime-result["total"]["step_time"])

        if result["total"]["max_qval"]-result["total"]["min_qval"]>2 and result["total"]["qval"]>0:
            q_puct = (result["total"]["max_qval"]-result["total"]["min_qval"])*result["total"]["qval"]*0.5
            result["total"]["q_puct"] += alpha * (q_puct-result["total"]["q_puct"])
            print("fix q_puct:", self.q_puct)
        else:
            result["total"]["q_puct"] = 1
        
        # 速度控制在消耗50行
        if win_values[0]==1:
            result["total"]["win_lost_tie"][0] += 1
        if win_values[1]==1:
            result["total"]["win_lost_tie"][1] += 1
        if win_values[0]==-1 and win_values[1]==-1:
            result["total"]["win_lost_tie"][2] += 1
        
        result["total"]["agent"] += 1
        result["total"]["_agent"] += 1

    
        if game_score>result["best"]["reward"]:
            result["best"]["reward"] = game_score
            result["best"]["agent"] = result["total"]["agent"]
        else:
            result["best"]["reward"] = round(result["best"]["reward"] - 0.9999,4)

        # 计算 acc 看有没有收敛
        pacc = []
        # vacc = []
        depth = []
        ns = []
        for m, data in enumerate([play_data[0]["data"], play_data[1]["data"]]) :
            for step in data["steps"]:
                pacc.append(step["acc_ps"])
                depth.append(step["depth"])
                ns.append(step["ns"])
                # vacc.append(step["state_value"])

        pacc = float(np.average(pacc))
        # vacc = float(np.average(vacc))
        vacc = abs(play_data[0]["agent"].piececount-play_data[1]["agent"].piececount)

        depth = float(np.average(depth))
        ns = float(np.average(ns))

        if result["total"]["pacc"]==0:
            result["total"]["pacc"] = pacc
        else:
            result["total"]["pacc"] = result["total"]["pacc"]*0.99 + pacc*0.01   

        if result["total"]["vacc"]==0:
            result["total"]["vacc"] = vacc
        else:
            result["total"]["vacc"] = result["total"]["vacc"]*0.99 + vacc*0.01   

        if result["total"]["depth"]==0:
            result["total"]["depth"] = depth
        else:
            result["total"]["depth"] = result["total"]["depth"]*0.99 + depth*0.01   

        if result["total"]["ns"]==0:
            result["total"]["ns"] = ns
        else:
            result["total"]["ns"] = result["total"]["ns"]*0.99 + ns*0.01   

        max_list_len=30
        if result["total"]["_agent"]>20:
            result["reward"].append(round(result["total"]["avg_score"],2))
            result["depth"].append(round(result["total"]["depth"],1))
            result["pacc"].append(round(result["total"]["pacc"],2))
            result["vacc"].append(round(result["total"]["vacc"],2))
            result["time"].append(round(result["total"]["step_time"],1))
            result["qval"].append(round(result["total"]["qval"],4))
            result["piececount"].append(round(result["total"]["avg_piececount"],1))
            result["piececount0_mcts"].append(round(result["total"]["piececount0_mcts"],1))
            result["piececount1_mcts"].append(round(result["total"]["piececount1_mcts"],1))
            result["q_puct"].append(round(result["total"]["q_puct"],2))
            
            local_time = time.localtime(start_time)
            current_month = local_time.tm_mon
            current_day = local_time.tm_mday

            result["update"].append(current_day)
            result["total"]["_agent"] -= 20 

            while len(result["reward"])>max_list_len:
                result["reward"].remove(result["reward"][0])
            while len(result["depth"])>max_list_len:
                result["depth"].remove(result["depth"][0])
            while len(result["pacc"])>max_list_len:
                result["pacc"].remove(result["pacc"][0])
            while len(result["vacc"])>max_list_len:
                result["vacc"].remove(result["vacc"][0])
            while len(result["time"])>max_list_len:
                result["time"].remove(result["time"][0])
            while len(result["qval"])>max_list_len:
                result["qval"].remove(result["qval"][0])
            while len(result["q_puct"])>max_list_len:
                result["q_puct"].remove(result["q_puct"][0])
            while len(result["piececount"])>max_list_len:
                result["piececount"].remove(result["piececount"][0])
            while len(result["piececount0_mcts"])>max_list_len:
                result["piececount0_mcts"].remove(result["piececount0_mcts"][0])
            while len(result["piececount1_mcts"])>max_list_len:
                result["piececount1_mcts"].remove(result["piececount1_mcts"][0])
            while len(result["update"])>max_list_len:
                result["update"].remove(result["update"][0])
            
            # 如果每步的消耗时间小于self.limit_steptime秒，增加探测深度    
            result["total"]["n_playout"] += round(self.limit_steptime-result["total"]["step_time"])
                
            # 保存下中间步骤的agent
            # newmodelfile = model_file+"_"+str(result["total"]["agent"])
            # if not os.path.exists(newmodelfile):
            #     policy_value_net.save_model(newmodelfile)

            # 如果当前最佳，将模型设置为最佳模型
            if max(result["reward"])==result["reward"][-1]:
                newmodelfile = model_file+"_reward_"+str(result["reward"][-1])
                if not os.path.exists(newmodelfile):
                    policy_value_net.save_model(newmodelfile)
                if os.path.exists(bestmodelfile): os.remove(bestmodelfile)
                if os.path.exists(newmodelfile): os.link(newmodelfile, bestmodelfile)
                        
        result["lastupdate"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.save_status_file(result, game_json) 
        
        # 全局评价
        values = [win_values[0]]*play_data[0]["agent"].steps + \
                        [win_values[1]]*play_data[1]["agent"].steps                        
        print("step_values:", values)
        
        # 局部奖励均值, 计算均值Q，用于 预测价值-均值Q 使其网络更平稳
        rewards = [play_data[0]["qval"]/play_data[0]["agent"].steps]*play_data[0]["agent"].steps + \
                        [play_data[1]["qval"]/play_data[1]["agent"].steps]*play_data[1]["agent"].steps                                    
        print("step_reward:", rewards)
            
        states, mcts_probs= [], []

        for data in [play_data[0]["data"], play_data[1]["data"]]:
            for step in data["steps"]:
                states.append(step["state"])
                mcts_probs.append(step["move_probs"])
                
        assert len(states)>0
        assert len(states)==len(values)
        assert len(states)==len(mcts_probs)
        assert len(states)==len(rewards)

        print("TRAIN Self Play end. length: %s value sum: %s saving ..." % (len(states),sum(values)))

        # 保存对抗数据到data_buffer
        filetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for i, obj in enumerate(self.get_equi_data(states, mcts_probs, values, rewards)):
        # for i, obj in enumerate(zip(states, mcts_probs, values, rewards)):
            filename = "{}-{}.pkl".format(filetime, i)
            savefile = os.path.join(data_wait_dir, filename)
            with open(savefile, "wb") as fn:
                pickle.dump(obj, fn)
        print("saved file basename:", filetime, "length:", i+1)
        print()
        play_data[0]["agent"].print()
        play_data[1]["agent"].print()
        print("agent 0 score:", play_data[0]["agent"].removedlines, "agent 1 score:", play_data[1]["agent"].removedlines)
        print("agent 0 steps:", play_data[0]["agent"].steps, "agent 1 steps:", play_data[1]["agent"].steps)
        print("agent 0 piececount:", play_data[0]["agent"].piececount, "agent 1 piececount:", play_data[1]["agent"].piececount)
        print("agent 0 paytime:", play_data[0]["paytime"], "agent 1 paytime:", play_data[1]["paytime"])
        print("max_game_qval:", max_game_qval, "min_game_qval:", min_game_qval, "avg_qval:", avg_qval)

    def run(self):
        """启动训练"""
        try:
            self.collect_selfplay_data()    
        except KeyboardInterrupt:
            print('quit')

def profiler():
    import cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    training = Train()
    training.run()
    profiler.disable()
    profiler.print_stats()

if __name__ == '__main__':
    print('start playing',datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    np.set_printoptions(precision=2, suppress=True)

    training = Train()
    training.run()
    # profiler()
    print('end playing',datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("")
