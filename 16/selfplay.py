import os, glob, pickle

from model import PolicyValueNet, data_dir, data_wait_dir, model_file
from agent_numba import Agent, ACTIONS
from mcts_single_numba import MCTSPlayer

import time, json, datetime

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
        self.n_playout = 128  # 每个动作的模拟战记录个数，影响后续 512/2 = 256；256/16 = 16个方块 的走法
        # 64/128/256/512 都不行
        # step -> score
        # 128  --> 0.7
        # self.n_playout = 128  # 每个动作的模拟战记录个数，影响后续 128/2 = 66；64/16 = 4个方块 的走法
        self.play_size = 20 # 每次测试次数
        self.buffer_size = 1000000  # cache对次数
        self.epochs = 2  # 每次更新策略价值网络的训练步骤数, 推荐是5
        self.kl_targ = 0.02  # 策略价值网络KL值目标
        self.best_win_ratio = 0.0
        
        # aplhazero 的最佳值是 4 
        # aplhatensor 是 5
        # MCTS child权重， 用来调节MCTS搜索深度，越大搜索越深，越相信概率，越小越相信Q 的程度 默认 5
        # 由于value完全用结果胜负来拟合，所以value不稳，只能靠概率p拟合，最后带动value来拟合
        self.c_puct = 5  

        self.max_step_count = 10000 

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
                ext = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                os.replace(status_file, status_file+"_"+ext) 
        if result==None:
            result={"reward":[], "depth":[], "pacc":[], "vacc":[], "time":[], "ns":[], "piececount":[]}
        if "total" not in result:
            result["total"]={"agent":0, "pacc":0, "vacc":0, "ns":0, "reward":0, "depth":0, "step_time":0, "_agent":0}
        if "best" not in result:
            result["best"]={"reward":0, "agent":0}
        if "piececount" not in result["total"]:
            result["total"]["piececount"]=0
        if "avg_piececount" not in result["total"]:
            result["total"]["avg_piececount"]=20            
        if "avg_reward_piececount" not in result["total"]:
            result["total"]["avg_reward_piececount"]=0            
        if "n_playout" not in result["total"]:
            result["total"]["n_playout"]=self.n_playout
        if "win_count" not in result["total"]:
            result["total"]["win_count"]=0            
        if "lost_count" not in result["total"]:
            result["total"]["lost_count"]=0           
        if "avg_score" not in result["total"]:
            result["total"]["avg_score"]=0  
        if "avg_score_ex" not in result["total"]:
            result["total"]["avg_score_ex"]=0  
        if "avg_qval" not in result["total"]:
            result["total"]["avg_qval"]=0  
        if "avg_state_value" not in result["total"]:
            result["total"]["avg_state_value"]=0  
        if "exrewardRate" not in result["total"]:
            result["total"]["exrewardRate"]=0  
        if "piececount" not in result:
            result["piececount"]=[]
        if "exrewardRate" not in result:
            result["exrewardRate"]={}
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
            if score==0 and mcts_prob[0]<0.1:
                equi_state = np.array([np.fliplr(s) for s in state])
                equi_mcts_prob = mcts_prob[[0,2,1,3]]
                extend_data.append((equi_state, equi_mcts_prob, value, score))
            # if i==0:
            #     print("state:",state)
            #     print("mcts_prob:",mcts_prob)
            #     print("equi_state:",equi_state)
            #     print("equi_mcts_prob:",equi_mcts_prob)
            #     print("value:",value)
        return extend_data

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

        # 游戏代理
        borads = []

        # 开始游戏
        policy_value_net = PolicyValueNet(GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=load_model_file)
        bestmodelfile = model_file+"_best"
        
        game_json = os.path.join(data_dir, "result.json")
  
        data = {"steps":[],"shapes":[],"last_state":0,"score":0,"piece_count":0}
        print('start game time:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        result = self.read_status_file(game_json)     

        # 先运行测试
        his_pieces = None
        min_score = result["total"]["avg_score"]
        for _ in range(self.play_size):
            result = self.read_status_file(game_json)     
            agent = Agent(isRandomNextPiece=True)
            start_time = time.time()
            agent.show_mcts_process= False
            agent.id = 0

            for i in range(self.max_step_count):
                action = policy_value_net.policy_value_fn_best_act(agent)
                _, reward = agent.step(action)
                if reward > 0:
                    print("#"*40, 'score:', agent.score, 'height:', agent.pieceheight, 'piece:', agent.piececount, "shape:", agent.fallpiece["shape"], \
                        'step:', agent.steps, "step time:", round((time.time()-start_time)/i,3),'avg_score:', result["total"]["avg_score"])            

                if agent.terminal:            
                    result["total"]["avg_score"] += (agent.removedlines-result["total"]["avg_score"])/1000
                    result["total"]["avg_piececount"] += (agent.piececount-result["total"]["avg_piececount"])/1000
                    result["lastupdate"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    break

            agent.print()

            # 判断是否需要重新玩,如果当前小于平均的0.75，放到运行池训练
            if agent.score < min_score:
                min_score = agent.score
                his_pieces = agent.piecehis
                # filename = "T{}-{}-{}.pkl".format(agent.piececount, agent.removedlines ,int(round(time.time() * 1000000)))
                # savefile = os.path.join(self.waitplaydir, filename)
                # with open(savefile, "wb") as fn:
                #     pickle.dump(his_pieces, fn)
                # print("save need replay", filename)
            
            self.save_status_file(result, game_json) 

        # must_reward_pieces_count = max(5,result["total"]["avg_reward_piececount"])
        # must_reward_pieces_count = min(10,must_reward_pieces_count)
        # 正式运行
        player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)

        # files = glob.glob(os.path.join(self.waitplaydir, "*.pkl"))
        # if len(files)>0:
        
        #     files = sorted(files, key=lambda x: os.path.getmtime(x))
        #     while len(files)>1000:
        #         fn = files.pop(0)
        #         os.remove(fn)

        #     his_pieces_file = files[0]
        #     try:
        #         with open(his_pieces_file,"rb") as fn:
        #             his_pieces = pickle.load(fn)     
        #         print("replay test again, load file:", his_pieces_file)
        #     finally:
        #         print("delete", his_pieces_file)
        #         os.remove(his_pieces_file)
        #     if not isinstance(his_pieces[0],str): his_pieces=[]                
        #     print([p for p in his_pieces])

        #     # 有历史记录的，按概率走，调整概率
        #     agent = Agent(isRandomNextPiece=False, nextPiecesList=his_pieces)
        #     agent.is_replay = True
        if his_pieces!=None:
            agent = Agent(isRandomNextPiece=False, nextPiecesList=his_pieces)
            agent.is_replay = True
        else:
            # 新局按Q值走，探索
            agent = Agent(isRandomNextPiece=False, )
            agent.is_replay = False

        agent.show_mcts_process= True
        agent.id = 0 if random.random()>0.5 else 1
        agent.exreward = True #random.random()>0.5
        exrewardRateKey="0.0"
        if agent.exreward:
            # v,p = [], np.ones((len(result["exrewardRate"])))
            # for i, k in enumerate(result["exrewardRate"]):
            #     v.append(k)
            #     p[i]=result["exrewardRate"][k] if result["exrewardRate"][k]>0 else 0.01
            # exrewardRateKey=np.random.choice(v, p=p/np.sum(p))
            # if random.random()>1/((sum(result["exrewardN"].values())+1)**0.5):
            # if agent.is_replay:
                # exrewardRateKey = random.choice(list(result["exrewardRate"].keys()))
                # Thompson Sampling
                # exrewardRateKeys = list(result["exrewardRate"].keys())
                # len_exrewardRate = len(exrewardRateKeys)
                # pbeta = [0 for _ in range(len_exrewardRate)]
                # for i in range(0, len_exrewardRate):
                #     win,trial = result["exrewardN"][exrewardRateKeys[i]]
                #     pbeta[i] = np.random.beta(win+1, trial-win+1)
                # choice = np.argmax(pbeta)
                # exrewardRateKey = exrewardRateKeys[choice]
                
            # else:
            # Thompson Sampling
            # exrewardRateKeys = list(result["exrewardRate"].keys())    
            # exrewardRateKey = min(exrewardRateKeys, key=lambda x:abs(float(x)-result["total"]["exrewardRate"]))
            exrewardRateKey = str(round(result["total"]["exrewardRate"], 1))
            
                # exrewardRateKey = "0.1"
                # exrewardRateKey = max(result["exrewardRate"], key=result["exrewardRate"].get)
            # else:
            # # elif random.random()>0.5:
            #     exrewardRateKeys = list(result["exrewardRate"].keys())
            #     len_exrewardRate = len(exrewardRateKeys)
            #     exrewardRateKey = exrewardRateKeys[np.random.randint(0, len_exrewardRate)]
            # else:
                  
            # exrewardRateKeys = list(result["exrewardRate"].keys())
            # len_exrewardRate = len(exrewardRateKeys)
            # pbeta = [0 for _ in range(len_exrewardRate)]
            # for i in range(0, len_exrewardRate):
            #     win,trial = result["exrewardN"][exrewardRateKeys[i]]
            #     pbeta[i] = np.random.beta(win+1, trial-win+1)
            # choice = np.argmax(pbeta)
            # exrewardRateKey = exrewardRateKeys[choice]
                
                # exrewardRateKey = max(result["exrewardRate"], key=result["exrewardRate"].get)
            agent.exrewardRate = result["total"]["exrewardRate"]  #float(exrewardRateKey)    
            
            train_conf_file=os.path.join(data_dir,"train_conf_pkl")
            if os.path.exists(train_conf_file):
                with open(train_conf_file, "rb") as fn:
                    train_conf = pickle.load(fn)
                    agent.exrewardRate = 1./train_conf["std_values"]
            
        else:
            agent.exrewardRate = 0
        agent.limitstep = random.random()<0.25
        max_emptyCount = random.randint(10,30)
        start_time = time.time()
        print("exreward:", agent.exreward,"exrewardRate:", agent.exrewardRate ,"max_emptyCount:",max_emptyCount,"isRandomNextPiece:",agent.isRandomNextPiece,"limitstep:",agent.limitstep)
        piececount = agent.piececount
        mark_reward_piececount = -1
        avg_qval=0
        avg_state_value=0
        for i in range(self.max_step_count):
            _step={"step":i, "curr_player":agent.id}
            _step["state"] = np.copy(agent.current_state())
            # print(_step["state"][0])           
            _step["piece_count"] = agent.piececount               
            _step["shape"] = agent.fallpiece["shape"]
            _step["pre_piece_height"] = agent.pieceheight

            action, qval, move_probs, state_value, max_qval, acc_ps, depth, ns = player.get_action(agent, agent.id, temp=1, avg_ns=result["total"]["ns"], avg_piececount=result["total"]["piececount"]) 

            _, reward = agent.step(action)

            if qval > 1:
                avg_qval += (1-state_value)**2
            elif qval < -1:
                avg_qval += (state_value-1)**2
            else:
                avg_qval += (qval-state_value)**2
            avg_state_value += state_value 

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

            time.sleep(0.1)

            # 这里的奖励是消除的行数
            if agent.state==1 and agent.last_reward>0:
                repeat_count = 40
                # print(_step["state"][0])
                # print(_step["state"][-1])
                print("#"*repeat_count, 'score:', agent.score, "reward:",reward, 'qval', round(qval,2), 'height:', agent.pieceheight, 'piece:', agent.piececount, \
                    'step:', agent.steps, "step time:", round((time.time()-start_time)/i,3),'reward_p:', agent.piececount-mark_reward_piececount)
                # if agent.score>result["total"]["reward"]+20: game_stop=True

            piececount = agent.piececount

            # 如果游戏结束或这一局已经超过了1小时
            paytime = time.time()-start_time
            if agent.terminal or (paytime>3600):
                data["score"] = agent.score
                data["piece_count"] = agent.piececount
                data["piece_height"] = agent.pieceheight
                borads.append(agent.board)                    

                game_score =  agent.removedlines 
                result = self.read_status_file(game_json)
                
                steptime = paytime/agent.steps
                
                avg_qval = avg_qval/agent.steps
                avg_state_value = avg_state_value/agent.steps
                
                print("step pay time:", steptime, "qval:", avg_qval, "avg_state_value:", avg_state_value)
                result["total"]["avg_score_ex"] += (game_score-result["total"]["avg_score_ex"])/100
                result["total"]["avg_reward_piececount"] += (game_score/agent.piececount - result["total"]["avg_reward_piececount"])/1000
                
                # delta = avg_qval - result["total"]["avg_qval"]
                # result["total"]["avg_qval"] += delta/100
                # delta2 = avg_qval - result["total"]["avg_qval"]
                # result["total"]["exrewardRate"] += (delta * delta2-result["total"]["exrewardRate"])/100                
                
                alpha = 0.01
                # _v = avg_qval - result["total"]["avg_qval"]
                result["total"]["avg_qval"] += alpha * (avg_qval - result["total"]["avg_qval"])
                result["total"]["avg_state_value"] += alpha * (avg_state_value - result["total"]["avg_state_value"])
                # result["total"]["exrewardRate"] = (1 - alpha) * (result["total"]["exrewardRate"] + alpha * avg_qval)
                result["total"]["exrewardRate"] = agent.exrewardRate

                mark_score = result["total"]["avg_score_ex"]

                # 速度控制在消耗50行
                if agent.score >= mark_score:
                    result["total"]["n_playout"] -= 1
                    result["total"]["win_count"] += 1
                else:
                    result["total"]["n_playout"] += 1
                    result["total"]["lost_count"] += 1
                
                result["total"]["agent"] += 1
                result["total"]["_agent"] += 1

                if result["total"]["step_time"]==0:
                    result["total"]["step_time"] = steptime
                else:
                    result["total"]["step_time"] += (steptime-result["total"]["step_time"])/100
            
                if game_score>result["best"]["reward"]:
                    result["best"]["reward"] = game_score
                    result["best"]["agent"] = result["total"]["agent"]
                else:
                    result["best"]["reward"] = round(result["best"]["reward"] - 0.9999,4)

                if result["total"]["reward"]==0:
                    result["total"]["reward"] = game_score
                else:
                    result["total"]["reward"] += (game_score-result["total"]["reward"])/100

                if result["total"]["piececount"]==0:
                    result["total"]["piececount"] = agent.piececount
                else:
                    result["total"]["piececount"] += (agent.piececount-result["total"]["piececount"])/100

                # 计算 acc 看有没有收敛
                pacc = []
                vacc = []
                depth = []
                ns = []
                # winner  = True if agent.piececount > result["total"]["piececount"] else False
                for step in data["steps"]:
                    pacc.append(step["acc_ps"])
                    depth.append(step["depth"])
                    ns.append(step["ns"])
                    vacc.append(step["qval"])
                    # if (not winner and step["state_value"]>0) or (winner and step["state_value"]<0):
                    #     vacc.append(0)
                    # else:
                    #     vacc.append(1)

                pacc = float(np.average(pacc))
                vacc = float(np.sum([1 for v in vacc if v>0]))
                depth = float(np.average(depth))
                ns = float(np.average(ns))

                if agent.exreward and not agent.is_replay:
                    if exrewardRateKey not in result["exrewardRate"]:
                        result["exrewardRate"][exrewardRateKey]=0
                    _q = result["exrewardRate"][exrewardRateKey]
                    result["exrewardRate"][exrewardRateKey] = round(_q+(agent.removedlines-_q)/10, 2)

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

                if result["total"]["_agent"]>100:
                    result["reward"].append(round(result["total"]["avg_score"],2))
                    result["depth"].append(round(result["total"]["depth"],1))
                    result["pacc"].append(round(result["total"]["pacc"],2))
                    result["vacc"].append(round(result["total"]["vacc"],2))
                    result["time"].append(round(result["total"]["step_time"],1))
                    result["ns"].append(round(result["total"]["avg_piececount"],1))
                    result["piececount"].append(round(result["total"]["piececount"],1))
                    result["total"]["_agent"] -= 100 

                    while len(result["reward"])>100:
                        result["reward"].remove(result["reward"][0])
                    while len(result["depth"])>100:
                        result["depth"].remove(result["depth"][0])
                    while len(result["pacc"])>100:
                        result["pacc"].remove(result["pacc"][0])
                    while len(result["vacc"])>100:
                        result["vacc"].remove(result["vacc"][0])
                    while len(result["time"])>100:
                        result["time"].remove(result["time"][0])
                    while len(result["ns"])>100:
                        result["ns"].remove(result["ns"][0])
                    while len(result["piececount"])>100:
                        result["piececount"].remove(result["piececount"][0])

                    # 保存下中间步骤的agent
                    newmodelfile = model_file+"_"+str(result["total"]["agent"])
                    if not os.path.exists(newmodelfile):
                        policy_value_net.save_model(newmodelfile)

                    # 如果当前最佳，将模型设置为最佳模型
                    if max(result["depth"])==result["depth"][-1]:
                        newmodelfile = model_file+"_depth_"+str(result["depth"][-1])
                        if not os.path.exists(newmodelfile):
                            policy_value_net.save_model(newmodelfile)
                        if os.path.exists(bestmodelfile): os.remove(bestmodelfile)
                        if os.path.exists(newmodelfile): os.link(newmodelfile, bestmodelfile)

                result["lastupdate"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.save_status_file(result, game_json) 
                break

        # 打印borad：
        for y in range(GAME_HEIGHT):
            line=""
            for b in borads:
                line+="| "
                for x in range(GAME_WIDTH):
                    if b[y][x]==0:
                        line+="  "
                    else:
                        line+="%s " % b[y][x]
            print(line)
        print((" "+" -"*GAME_WIDTH+" ")*len(borads))

        # winner = True if agent.piececount > result["total"]["piececount"] else False

        # print("winner: %s piececount: %s %s" %(winner, agent.piececount, result["total"]["piececount"]))

        # 更新reward和score，reward为胜负，[1|-1|0]；score 为本步骤以后一共消除的行数
        step_count = len(data["steps"])
       
        # 奖励的分配
        # 重新定义价值为探索深度，因此value应该是[0,-X]
        # 如何评价最后的得分？最完美的情况（填满比）

        # 奖励的位置
        piececount = data["steps"][-1]["piece_count"]+1
        pieces_reward = [0 for _ in range(piececount)]
        pieces_steps = [0 for _ in range(piececount)]

        # # 统计所有获得奖励的方块
        for m in range(step_count):
            pieces_steps[data["steps"][m]["piece_count"]] = m
            if data["steps"][m]["reward"]>0:
                pieces_reward[data["steps"][m]["piece_count"]] = data["steps"][m]["reward"] 

        pieces_value = [round(data["steps"][pieces_steps[p]]["qval"],2) for p in range(piececount)]
        pieces_probs = [round(np.max(data["steps"][pieces_steps[p]]["move_probs"]),2) for p in range(piececount)]

        print()
        print("reward:", pieces_reward)
        print()
        print("value: ", pieces_value)
        print()
        print("probs: ", pieces_probs)
        print()

        print(i,"score:",data["score"],"piece_count:",data["piece_count"],"piece_height:",data["piece_height"],"steps:",step_count)
       
        states, mcts_probs, values, scores= [], [], [], []

        for step in data["steps"]:
            states.append(step["state"])
            mcts_probs.append(step["move_probs"])
            values.append(step["qval"])
            scores.append(step["acc_ps"])
                
        # print([round(v,2) for v in values])
        # print([round(s,2) for s in scores])
        
        
        assert len(states)>0
        assert len(states)==len(values)
        assert len(states)==len(mcts_probs)
        assert len(states)==len(scores)

        print("TRAIN Self Play end. length: %s value sum: %s saving ..." % (len(states),sum(values)))

        # 保存对抗数据到data_buffer
        filetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # 现在第一层改为了横向，所以不能做图片左右翻转增强
        for i, obj in enumerate(self.get_equi_data(states, mcts_probs, values, scores)):
        # for i, obj in enumerate(zip(states, mcts_probs, values, score)):
            filename = "{}-{}.pkl".format(filetime, i)
            savefile = os.path.join(data_wait_dir, filename)
            with open(savefile, "wb") as fn:
                pickle.dump(obj, fn)
        print("saved file basename:", filetime, "length:", i+1)

        # 删除训练集
        if agent.piececount/result["total"]["piececount"]<0.5:
            filename = "R{}-{}-{}.pkl".format(agent.piececount, agent.score, int(round(time.time() * 1000000)))
            his_pieces_file = os.path.join(self.waitplaydir, filename)
            print("save need replay", his_pieces_file)
            with open(his_pieces_file, "wb") as fn:
                pickle.dump(agent.piecehis, fn)

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
    print('start training',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    training = Train()
    training.run()
    # profiler()
    print('end training',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("")

