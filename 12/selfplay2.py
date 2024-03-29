import os, pickle

from model import PolicyValueNet, data_dir, data_wait_dir, model_file
from agent import Agent, ACTIONS
from mcts import MCTSPlayer

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
        self.n_playout = 64  # 每个动作的模拟战记录个数，不能用32，因为收敛太慢了
        self.play_batch_size = 5 # 每次自学习次数
        self.buffer_size = 1000000  # cache对次数
        self.epochs = 2  # 每次更新策略价值网络的训练步骤数, 推荐是5
        self.kl_targ = 0.02  # 策略价值网络KL值目标
        self.best_win_ratio = 0.0
        
        self.c_puct = 1  # MCTS child权重， 用来调节MCTS中 探索/乐观 的程度 默认 5


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
            result={"agent":0, "reward":[], "pieces":[], "qvals":[], "QVal":0}
        if "curr" not in result:
            result["curr"]={"reward":0, "pieces":0, "agent1000":0, "agent100":0, "height":0}
        if "best" not in result:
            result["best"]={"reward":0,"pieces":0,"agent":0}
        if "cpuct" not in result:
            result["cpuct"]={"0.5":{"count":0,"value":0},"1.5":{"count":0,"value":0}}    
        for key in result["cpuct"]:
            if not isinstance(result["cpuct"][key], dict):
                result["cpuct"][key] = {"count":0,"value":result["cpuct"][key]}
        if "time" not in result:
            result["time"]={"agent_time":0,"step_time":0,"step_times":[]}
        if "height" not in result:
            result["height"]=[]
        if "height" not in result["curr"]:
            result["curr"]["height"]=0
        if "vars" not in result or "avg" not in result["vars"]:
            # 缺省std = sqrt(2)
            result["vars"]={"max":1, "min":-1, "std":1, "avg":0}
        if "shapes" not in result:
            result["shapes"]={"t":0, "i":0, "j":0, "l":0, "s":0, "z":0, "o":0}
        if "first_reward" not in result:
            result["first_reward"]=0
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
            equi_state = np.array([np.fliplr(s) for s in state])
            equi_mcts_prob = mcts_prob[[0,1,3,2,4]]
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

        jsonfile = os.path.join(data_dir, "result.json")

        # 游戏代理
        agent = Agent()

        max_game_num = 2
        agentcount, agentreward, piececount, agentscore = 0, 0, 0, 0

        borads = []
        game_num = 0       

        # 尽量不要出现一样的局面
        game_keys = []
        game_datas = []
        # 开始一局游戏
        policy_value_net = PolicyValueNet(GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=model_file)
        for _ in count():
            start_time = time.time()
            game_num += 1
            print('start game :', game_num, 'time:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

            game_flip_v = game_num%2==0
            if game_flip_v:
                jsonfile = os.path.join(data_dir, "result_flip_v.json")
            else:
                jsonfile = os.path.join(data_dir, "result.json")

            result = self.read_status_file(jsonfile)
            print("QVal:",result["QVal"])

            # c_puct 参数自动调节，step=0.1 
            cpuct_result = result["cpuct"]

            cpuct_list = sorted(cpuct_result, key=lambda x : cpuct_result[x]["count"])
            print("cpuct:",cpuct_result, "-->", cpuct_list)
            cpuct = float(cpuct_list[0])

            print("c_puct:",cpuct, "n_playout:",self.n_playout)

            player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=cpuct, n_playout=self.n_playout, flip_v=game_flip_v)
  
            _data = {"steps":[],"shapes":[],"last_state":0,"score":0,"piece_count":0}
            game = copy.deepcopy(agent)
            # game = Agent(isRandomNextPiece=False)

            if game_num==1 or game_num==max_game_num:
                game.show_mcts_process=True

            piece_idx = []

            for i in count():
                _step={"step":i}
                _step["state"] = game.current_state()               
                _step["piece_count"] = game.piececount               
                _step["shape"] = game.fallpiece["shape"]
                _step["pre_piece_height"] = game.pieceheight

                if game_num == 1:
                    action, move_probs, state_value = player.get_action(game, temp=1/(game.pieceheight+1)) 
                else: 
                    action, move_probs, state_value = player.get_action(game, temp=1/(game.pieceheight+1)) 

                    if game.get_key() in game_keys:
                        print(game.steps, game.piececount, game.fallpiece["shape"], game.piecesteps, "key:", game.get_key(), "key_len:" ,len(game_keys))
                        action = random.choice(game.get_availables())

                # 如果当前选择的啥也不做 KEY_NONE， 不过 KEY_DOWN 也可以用时，有一半几率直接用 KEY_DOWN
                # if action == ACTIONS[0] and random.random()>0.5 and ACTIONS[-1] in game.get_availables():
                #     action = ACTIONS[-1]

                _, reward = game.step(action)

                _step["piece_height"] = game.pieceheight

                _step["key"] = game.get_key()
                # 这里不鼓励多行消除
                _step["reward"] = reward if reward>0 else 0
                _step["action"] = action                
                _step["move_probs"] = move_probs
                _step["state_value"] = state_value
                _data["shapes"].append(_step["shape"])
                _data["steps"].append(_step)

                # 这里的奖励是消除的行数
                if reward > 0:
                    result = self.read_status_file(jsonfile)
                    if result["curr"]["height"]==0:
                        result["curr"]["height"]=game.pieceheight
                    else:
                        result["curr"]["height"] = round(result["curr"]["height"]*0.99 + game.pieceheight*0.01, 2)
                    result["shapes"][_step["shape"]] += reward

                    # 如果是第一次奖励，记录当前的是第几个方块
                    if game.score==reward:
                        if result["first_reward"]==0:
                            result["first_reward"]=game.piececount
                        else:
                            result["first_reward"]=result["first_reward"]*0.99 + game.piececount*0.01

                        # 如果第一次的奖励低于平均数，则将前面的几个方块也进行奖励
                        # if game.piececount < result["first_reward"]:
                        #     for idx in piece_idx:
                        #         _data["steps"][idx]["reward"]=0.5

                    self.save_status_file(result, jsonfile)
                    print("#"*40, 'score:', game.score, 'height:', game.pieceheight, 'piece:', game.piececount, "shape:", game.fallpiece["shape"], \
                        'step:', i, "step time:", round((time.time()-start_time)/i,3),'flip:', game_flip_v, "#"*40)

                # 记录当前的方块放置的 idx
                if game.state != 0:
                    piece_idx.append(i)

                # 方块的个数越多越好
                if game.terminal :
                    _data["score"] = game.score
                    _data["piece_count"] = game.piececount

                    # 更新状态
                    game_reward =  game.score   

                    result = self.read_status_file(jsonfile)
                    if result["QVal"]==0:
                        result["QVal"] = game_reward
                    else:
                        result["QVal"] = result["QVal"]*0.999 + game_reward*0.001   
                    paytime = time.time()-start_time
                    steptime = paytime/game.steps
                    if result["time"]["agent_time"]==0:
                        result["time"]["agent_time"] = paytime
                        result["time"]["step_time"] = steptime
                    else:
                        result["time"]["agent_time"] = round(result["time"]["agent_time"]*0.99+paytime*0.01, 3)
                        d = game.steps/10000.0
                        if d>1 : d = 0.99
                        result["time"]["step_time"] = round(result["time"]["step_time"]*(1-d)+steptime*d, 3)
                
                    # 记录当前cpuct的统计结果
                    cpuct_str = str(cpuct)
                    if cpuct_str in result["cpuct"]:
                        result["cpuct"][cpuct_str]["value"] = result["cpuct"][cpuct_str]["value"]+game_reward         
                        result["cpuct"][cpuct_str]["count"] = result["cpuct"][cpuct_str]["count"]+1         

                    if game_reward>result["best"]["reward"]:
                        result["best"]["reward"] = game_reward
                        result["best"]["pieces"] = game.piececount
                        result["best"]["score"] = game.score
                        result["best"]["agent"] = result["agent"]+agentcount

                    result["agent"] += 1
                    result["curr"]["reward"] += game.score
                    result["curr"]["pieces"] += game.piececount
                    result["curr"]["agent1000"] += 1
                    result["curr"]["agent100"] += 1

                    if result["curr"]["agent100"]>50:
                        result["reward"].append(round(result["curr"]["reward"]/result["curr"]["agent1000"],2))
                        result["pieces"].append(round(result["curr"]["pieces"]/result["curr"]["agent1000"],2))
                        result["qvals"].append(round(result["QVal"],2))
                        result["height"].append(result["curr"]["height"])
                        result["time"]["step_times"].append(result["time"]["step_time"])
                        result["curr"]["agent100"] -= 50 
                        while len(result["reward"])>200:
                            result["reward"].remove(result["reward"][0])
                        while len(result["pieces"])>200:
                            result["pieces"].remove(result["pieces"][0])
                        while len(result["qvals"])>200:
                            result["qvals"].remove(result["qvals"][0])
                        while len(result["height"])>200:    
                            result["height"].remove(result["height"][0])
                        while len(result["time"]["step_times"])>200:    
                            result["time"]["step_times"].remove(result["time"]["step_times"][0])

                        # 每100局更新一次cpuct参数
                        # qval = result["QVal"]
                        # cpuct表示概率的可信度
                        cpuct_list.sort()
                        v0 = result["cpuct"][cpuct_list[0]]["value"]/result["cpuct"][cpuct_list[0]]["count"]
                        v1 = result["cpuct"][cpuct_list[1]]["value"]/result["cpuct"][cpuct_list[1]]["count"]
                        if v0 > v1:
                            cpuct = round(float(cpuct_list[0])-0.1,1)
                            if cpuct<=0.1:
                                result["cpuct"] = {"0.1":{"count":0,"value":0}, "1.1":{"count":0,"value":0}}
                            else:
                                result["cpuct"] = {str(cpuct):{"count":0,"value":0}, str(round(cpuct+1,2)):{"count":0,"value":0}}
                        else:
                            cpuct = round(float(cpuct_list[0])+0.1,1)
                            result["cpuct"] = {str(cpuct):{"count":0,"value":0}, str(round(cpuct+1,2)):{"count":0,"value":0}}

                        if max(result["reward"])==result["reward"][-1]:
                            newmodelfile = model_file+"_reward_"+str(result["reward"][-1])
                            if not os.path.exists(newmodelfile):
                                policy_value_net.save_model(newmodelfile)

                    if result["curr"]["agent1000"]>1000:
                        result["curr"]={"reward":0,"pieces":0,"agent1000":0,"agent100":0,"height":0}

                        newmodelfile = model_file+"_"+str(result["agent"])
                        if not os.path.exists(newmodelfile):
                            policy_value_net.save_model(newmodelfile)
                    result["lastupdate"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    self.save_status_file(result, jsonfile) 

                    game.print()
                    print(game_num, 'reward:', game.score, "Qval:", game_reward, 'len:', i, "piececount:", game.piececount, "time:", time.time()-start_time)
                    print("pay:", time.time() - start_time , "s\n" )
                    agentcount += 1
                    agentscore += game.score
                    agentreward += game_reward
                    piececount += game.piececount



                    break

            for step in _data["steps"]:
                if not step["key"] in game_keys:                            
                    game_keys.append(step["key"])

            game_datas.append(_data)

            borads.append(game.board)

            # 如果训练样本超过10000，则停止训练
            if len(game_keys)> 10000: break

            # 如果训练次数超过了最大次数，则直接终止训练
            if game_num >= max_game_num: break

        # 打印borad：
        from game import blank 
        for y in range(agent.height):
            line=""
            for b in borads:
                line+="| "
                for x in range(agent.width):
                    if b[x][y]==blank:
                        line+="  "
                    else:
                        line+="%s " % b[x][y]
            print(line)
        print((" "+" -"*agent.width+" ")*len(borads))

        scores = [data["score"] for data in game_datas]
        max_score = max(scores)

        ## 放弃 按0.99的衰减更新reward
        # 只关注最后一次得分方块的所有步骤,将消行方块的所有步骤的得分都设置为1
        for data in game_datas:
            step_count = len(data["steps"])
            piece_count = -1
            v = 0
            score = 0
            vlist=[]
            slist=[]
            v_sum = 0
            s_sum = 0
            for i in range(step_count-1,-1,-1):
                # v = 0.99*v+data["steps"][i]["pre_piece_height"]-data["steps"][i]["piece_height"]
                # v = math.tanh(v)
                v =  data["steps"][i]['state_value']
                if data["score"]<max_score: v = v * -1.
                if piece_count!=data["steps"][i]["piece_count"]:
                    piece_count = data["steps"][i]["piece_count"]
                    score += data["steps"][i]["reward"]
                    vlist.insert(0,v)
                    slist.insert(0, score)
                data["steps"][i]["reward"] = v
                data["steps"][i]["score"] = score
                v_sum += v
                s_sum += score

            print("score:","avg",s_sum/step_count, slist)
            print("value:","avg",v_sum/step_count, vlist)
        # 总得分为 消行奖励  + (本局消行奖励-平均每局消行奖励/平均每局消行奖励)
        # for data in game_datas:
        #     step_count = len(data["steps"])
        #     weight = (data["score"]-result["QVal"])/result["QVal"]
        #     for i in range(step_count):
        #         # if data["steps"][i]["reward"] < 1:
        #         v = data["steps"][i]["reward"] + weight 
        #             # if v>1: v=1
        #         data["steps"][i]["reward"] = v
        
        # print("fixed reward")
        # for data in game_datas:
        #     step_count = len(data["steps"])
        #     piece_count = -1
        #     vlist=[]
        #     for i in range(step_count):
        #         if piece_count!=data["steps"][i]["piece_count"]:
        #             piece_count = data["steps"][i]["piece_count"]
        #             vlist.append(data["steps"][i]["reward"])
        #     print("score:", data["score"], "piece_count:", data["piece_count"],  [round(num, 2) for num in vlist])

        # 状态    概率      本步表现 本局奖励
        states, mcts_probs, values, score= [], [], [], []

        for data in game_datas:
            for step in data["steps"]:
                states.append(step["state"])
                mcts_probs.append(step["move_probs"])
                values.append(step["reward"])
                score.append(step["score"])

        # # 用于统计shape的std
        # pieces_idx={"t":[], "i":[], "j":[], "l":[], "s":[], "z":[], "o":[]}

        # var_keys = set()

        # for data in game_datas:
        #     for shape in set(data["shapes"]):
        #         var_keys.add(shape)
        # step_key_name = "shape"

        # for key in var_keys:
        #     _states, _mcts_probs, _values = [], [], []
        #     # _pieces_idx={"t":[], "i":[], "j":[], "l":[], "s":[], "z":[], "o":[]}
        #     for data in game_datas:
        #         for step in data["steps"]:
        #             if step[step_key_name]!=key: continue
        #             _states.append(step["state"])
        #             _mcts_probs.append(step["move_probs"])
        #             _values.append(step["reward"])
        #             # _pieces_idx[step["shape"]].append(len(values)+len(_values)-1)

        #     if len(_values)==0: continue
                
        #     # 重新计算
        #     curr_avg_value = sum(_values)/len(_values)
        #     curr_std_value = np.std(_values)
        #     if curr_std_value<0.01: continue

        #     # for shape in _pieces_idx:
        #     #     pieces_idx[shape].extend(_pieces_idx[shape])

        #     _normalize_vals = []
        #     # 用正态分布的方式重新计算
        #     curr_std_value_fix = curr_std_value + 1e-8 # * (2.0**0.5) # curr_std_value / result["vars"]["std"] 
        #     for v in _values:
        #         #标准化的标准差为 (x-μ)/(σ/std), std 为 1 # 1/sqrt(2)
        #         _nv = (v-curr_avg_value)/curr_std_value_fix 
        #         if _nv <-1 : _nv = -1
        #         if _nv >1  : _nv = 1
        #         if _nv == 0: _nv = 1e-8
        #         _normalize_vals.append(_nv)

        #     # 将最好的一步的值设置为1
        #     # max_normalize_val = max(_normalize_vals)-1
        #     # for i in range(len(_normalize_vals)):
        #     #     _normalize_vals[i] -= max_normalize_val

        #     print(key, len(_normalize_vals), "max:", max(_normalize_vals), "min:", min(_normalize_vals), "std:", curr_std_value)

        #     states.extend(_states)
        #     mcts_probs.extend(_mcts_probs)
        #     values.extend(_normalize_vals)
        #     result["vars"]["max"] = result["vars"]["max"]*0.999 + max(_normalize_vals)*0.001
        #     result["vars"]["min"] = result["vars"]["min"]*0.999 + min(_normalize_vals)*0.001
        #     result["vars"]["avg"] = result["vars"]["avg"]*0.999 + np.average(_normalize_vals)*0.001
        #     result["vars"]["std"] = result["vars"]["std"]*0.999 + np.std(_normalize_vals)*0.001
        #     # _states, _mcts_probs, _values = [], [], []

        # # if result["vars"]["max"]>1 or result["vars"]["min"]<-1:
        # #     result["vars"]["std"] = round(result["vars"]["std"]-0.0001,4)
        # # else:
        # #     result["vars"]["std"] = round(result["vars"]["std"]+0.0001,4)

        # json.dump(result, open(jsonfile,"w"), ensure_ascii=False)

        assert len(states)>0
        assert len(states)==len(values)
        assert len(states)==len(mcts_probs)
        assert len(states)==len(score)

        print("TRAIN Self Play end. length:%s value sum:%s saving ..." % (len(states),sum(values)))

        # 保存对抗数据到data_buffer
        for obj in self.get_equi_data(states, mcts_probs, values, score):
            filename = "{}.pkl".format(uuid.uuid1())
            savefile = os.path.join(data_wait_dir, filename)
            with open(savefile, "wb") as fn:
                pickle.dump(obj, fn)


        # 打印shape的标准差
        # for shape in pieces_idx:
        #     test_data=[]
        #     for i in pieces_idx[shape]:
        #         if i>=(len(values)): break
        #         test_data.append(values[i])
        #     if len(test_data)==0: continue
        #     print(shape, "len:", len(test_data), "max:", max(test_data), "min:", min(test_data), "std:", np.std(test_data))


 

    def run(self):
        """启动训练"""
        try:
            self.collect_selfplay_data()    
        except KeyboardInterrupt:
            print('quit')

if __name__ == '__main__':
    print('start training',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    training = Train()
    training.run()
    print('end training',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("")

