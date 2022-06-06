import os, glob, pickle

from model import PolicyValueNet, data_dir, data_wait_dir, model_file
from agent import Agent, ACTIONS
from mcts import MCTSPlayer

import sys, time, json, datetime

from itertools import count
from collections import deque
from collections import namedtuple
import os, math, random, uuid

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
        self.policy_value_net = PolicyValueNet(GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=model_file)

    def read_status_file(self, status_file):
        # 获取历史训练数据
        if os.path.exists(status_file):
            result=json.load(open(status_file,"r"))
        else:
            result={}
            result={"agent":0, "reward":[], "pieces":[], "qvals":[], "QVal":0}
        if "curr" not in result:
            result["curr"]={"reward":0, "pieces":0, "agent1000":0, "agent100":0, "height":0}
        if "best" not in result:
            result["best"]={"reward":0,"pieces":0,"agent":0}
        if "cpuct" not in result:
            result["cpuct"]={"0.1":0,"0.2":0}
        if "avg_time" not in result:
            result["avg_time"]=0
        if "height" not in result:
            result["height"]=[]
        if "height" not in result["curr"]:
            result["curr"]["height"]=0
        if "vars" not in result or "std" not in result["vars"]:
            # 缺省std = sqrt(2)
            result["vars"]={"max":1,"min":-1,"std":0.7}
        return result

    def collect_selfplay_data(self):
        """收集自我对抗数据用于训练"""       
        print("TRAIN Self Play starting ...")

        jsonfile = os.path.join(data_dir, "result.json")

        # 游戏代理
        agent = Agent()

        min_game_num = 5
        max_game_num = 7
        agentcount, agentreward, piececount, agentscore = 0, 0, 0, 0

        borads = []
        game_num = 0
        can_exit_flag = False
        
        cpuct_first_flag = random.random() > 0.5

        # 尽量不要出现一样的局面
        game_keys = []
        game_datas = []
        # 开始一局游戏
        for _ in count():
            start_time = time.time()
            game_num += 1

            result = self.read_status_file(jsonfile)
            print("QVal:",result["QVal"])

            # c_puct 参数自动调节，step=0.1 
            cpuct_list = []  
            for cp in result["cpuct"]:
                cpuct_list.append(cp)
                if len(cpuct_list)==2:break
            cpuct_list.sort()

            # fix error need remove
            if abs(result["cpuct"][cpuct_list[0]]-result["cpuct"][cpuct_list[1]])>50:
                result["cpuct"][cpuct_list[0]]=result["QVal"]
                result["cpuct"][cpuct_list[1]]=result["QVal"]
                json.dump(result, open(jsonfile,"w"), ensure_ascii=False)

            print("cpuct:",result["cpuct"])

            if cpuct_first_flag:
                cpuct = float(cpuct_list[0])
            else:
                cpuct = float(cpuct_list[1])
            cpuct_first_flag = not cpuct_first_flag

            print("game_num",game_num,"c_puct:",cpuct,"n_playout:",self.n_playout)
            player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=cpuct, n_playout=self.n_playout)

            _data = {"steps":[],"last_state":0,"score":0,"piece_count":0}
            game = copy.deepcopy(agent)
            # game = Agent()

            if game_num==1 or game_num==max_game_num:
                game.show_mcts_process=True

            for i in count():
                _step={"step":i}
                _step["state"] = game.current_state()               
                _step["piece_count"] = game.piececount               
                                
                if game_num == max_game_num:
                    action, move_probs = player.get_action(game, temp=self.temp, return_prob=1, need_random=False) 
                else: 
                    action, move_probs = player.get_action(game, temp=self.temp, return_prob=1, need_random=True, game_keys=game_keys) 

                _, reward = game.step(action)

                _step["key"] = game.get_key()
                _step["reward"] = reward
                _step["action"] = action                
                _step["move_probs"] = move_probs

                _data["steps"].append(_step)

                # 这里的奖励是消除的行数
                if reward > 0:
                    result = self.read_status_file(jsonfile)
                    if result["curr"]["height"]==0:
                        result["curr"]["height"]=game.pieceheight
                    else:
                        result["curr"]["height"] = round(result["curr"]["height"]*0.99 + game.pieceheight*0.01, 2)
                    json.dump(result, open(jsonfile,"w"), ensure_ascii=False)
                    print("#"*40, 'score:', game.score, 'height:', game.pieceheight, 'piece:', game.piececount, 'step:', i, "#"*40)

                # 方块的个数越多越好
                if game.terminal:
                    _game_last_reward = game.getNoEmptyCount()/200.
                    _data["reward"] = _game_last_reward
                    _data["score"] = game.score
                    _data["piece_count"] = game.piececount

                    # 更新状态
                    game_reward =  _game_last_reward + game.score   

                    result = self.read_status_file(jsonfile)
                    if result["QVal"]==0:
                        result["QVal"] = game_reward
                        result["avg_time"]= time.time()-start_time
                    else:
                        result["QVal"] = result["QVal"]*0.999 + game_reward*0.001   
                        result["avg_time"]= result["avg_time"]*0.999 + (time.time()-start_time)*0.001 
                    if game_reward > result["QVal"] and game.score>0: can_exit_flag = True
                   
                    # 记录当前cpuct的统计结果
                    if str(cpuct) in result["cpuct"]:
                        result["cpuct"][str(cpuct)] = result["cpuct"][str(cpuct)]*0.99 + game_reward*0.01         

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
                    json.dump(result, open(jsonfile,"w"), ensure_ascii=False) 

                    game.print()
                    print(game_num, 'reward:', game.score, "Qval:", game_reward, 'len:', i, "piececount:", game.piececount, "time:", time.time()-start_time)
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

            # 如果训练次数超过了最大次数，并且最大得分值超过了平均得分值，则停止训练
            if game_num >= min_game_num and can_exit_flag: break

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

        # 按0.50的衰减更新reward
        for data in game_datas:
            step_count = len(data["steps"])
            piece_count = -1
            v = 0
            for i in range(step_count-1,-1,-1):
                if piece_count!=data["steps"][i]["piece_count"]:
                    piece_count = data["steps"][i]["piece_count"]
                    v = 0.5*v+data["steps"][i]["reward"]
                data["steps"][i]["reward"] = v

        # 将所有的reward加上每一局的最后基础得分
        for data in game_datas:
            step_count = len(data["steps"])
            for i in range(step_count):
                data["steps"][i]["reward"] += data["reward"]
        
        states, mcts_probs, values= [], [], []

        # 分离每一步的全部步骤做同比
        max_piece_count = 0
        for data in game_datas:
            if data["piece_count"]>max_piece_count:
                max_piece_count = data["piece_count"]

        for p in range(max_piece_count):
            _info = []
            _states, _mcts_probs, _values = [], [], []
            for data in game_datas:
                for step in data["steps"]:
                    if step["piece_count"]!=p: continue
                    _states.append(step["state"])
                    _mcts_probs.append(step["move_probs"])
                    _values.append(step["reward"])
                if len(_values)==0:
                    _info.append(-1)
                else:      
                    _info.append(_values[-1])
            print(p, _info)

            if len(_states)==0: continue
                
            # 重新计算
            curr_avg_value = sum(_values)/len(_values)
            curr_std_value = np.std(_values)
            # 数据的标准差太小，则继续增加样本数量
            if curr_std_value<=0.1:
                print(p, "std too small:", len(_states), "std:", curr_std_value, _values[:3], "...", _values[-3:])  
                continue

            _normalize_vals = []
            curr_std_value_fix = curr_std_value / result["vars"]["std"] 
            for v in _values:
                #标准化的标准差为 (x-μ)/(σ/std), std 调整的规则是平均最大值和平均最小值都在 [-1 ~ 1] 的范围内
                _nv = (v-curr_avg_value)/curr_std_value_fix 
                if _nv == 0: _nv = 1e-8
                _normalize_vals.append(_nv)

            states.extend(_states)
            mcts_probs.extend(_mcts_probs)
            values.extend(_normalize_vals)
            print(p, len(_states),"std:", curr_std_value,  _normalize_vals[:3], "..." ,_normalize_vals[-3:])
            result["vars"]["max"] = result["vars"]["max"]*0.999 + max(_normalize_vals)*0.001
            result["vars"]["min"] = result["vars"]["min"]*0.999 + min(_normalize_vals)*0.001

            # _states, _mcts_probs, _values = [], [], []

        if result["vars"]["max"]>1 or result["vars"]["min"]<-1:
            result["vars"]["std"] = round(result["vars"]["std"]-0.0001,4)
        else:
            result["vars"]["std"] = round(result["vars"]["std"]+0.0001,4)

        json.dump(result, open(jsonfile,"w"), ensure_ascii=False)

        assert len(states)>0
        assert len(states)==len(values)
        assert len(states)==len(mcts_probs)

        print("TRAIN Self Play end. length:%s value sum:%s saving ..." % (len(states),sum(values)))

        # 保存对抗数据到data_buffer
        for obj in zip(states, mcts_probs, values):
            filename = "{}.pkl".format(uuid.uuid1())
            savefile = os.path.join(data_wait_dir, filename)
            pickle.dump(obj, open(savefile, "wb"))
       
        result = self.read_status_file(jsonfile)
        if result["curr"]["agent100"]>100:
            result["reward"].append(round(result["curr"]["reward"]/result["curr"]["agent1000"],2))
            result["pieces"].append(round(result["curr"]["pieces"]/result["curr"]["agent1000"],2))
            result["qvals"].append(round(result["QVal"],2))
            result["height"].append(result["curr"]["height"])
            result["curr"]["agent100"] -= 100 
            while len(result["reward"])>200:
                result["reward"].remove(result["reward"][0])
            while len(result["pieces"])>200:
                result["pieces"].remove(result["pieces"][0])
            while len(result["qvals"])>200:
                result["qvals"].remove(result["qvals"][0])
            while len(result["height"])>200:    
                result["height"].remove(result["height"][0])

            # 每100局更新一次cpuct参数
            qval = result["QVal"]
            if result["cpuct"][cpuct_list[0]]>result["cpuct"][cpuct_list[1]]:
                cpuct = round(float(cpuct_list[0])-0.01,2)
                if cpuct<=0.01:
                    result["cpuct"] = {"0.01":qval, "0.11":qval}
                else:
                    result["cpuct"] = {str(cpuct):qval, str(round(cpuct+0.1,2)):qval}
            else:
                cpuct = round(float(cpuct_list[1])+0.01,2)
                if cpuct<=0.11:
                    result["cpuct"] = {"0.01":qval, "0.11":qval}
                else:
                    result["cpuct"] = {str(round(cpuct-0.1,2)):qval, str(cpuct):qval}

            if max(result["reward"])==result["reward"][-1]:
                newmodelfile = model_file+"_reward_"+str(result["reward"][-1])
                if not os.path.exists(newmodelfile):
                    self.policy_value_net.save_model(newmodelfile)

        if result["curr"]["agent1000"]>1000:
            result["curr"]={"reward":0,"pieces":0,"agent1000":0,"agent100":0,"height":0}

            newmodelfile = model_file+"_"+str(result["agent"])
            if not os.path.exists(newmodelfile):
                self.policy_value_net.save_model(newmodelfile)
        result["lastupdate"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        json.dump(result, open(jsonfile,"w"), ensure_ascii=False)

    def policy_update(self, sample_data, epochs=1):
        """更新策略价值网络policy-value"""
        # 训练策略价值网络
        # 随机抽取data_buffer中的对抗数据
        # mini_batch = self.dataset.loadData(sample_data)
        state_batch, mcts_probs_batch, winner_batch = sample_data
        # # for x in mini_batch:
        # #     print("-----------------")
        # #     print(x)
        # # state_batch = [data[0] for data in mini_batch]
        # # mcts_probs_batch = [data[1] for data in mini_batch]
        # # winner_batch = [data[2] for data in mini_batch]

        # print(state_batch)

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)  
        for i in range(epochs):
            loss, v_loss, p_loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)

            # 散度计算：
            # D(P||Q) = sum( pi * log( pi / qi) ) = sum( pi * (log(pi) - log(qi)) )
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:  # 如果D_KL跑偏则尽早停止
                break

        # 自动调整学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        # 如果学习到了，explained_var 应该趋近于 1，如果没有学习到也就是胜率都为很小值时，则为 0
        explained_var_old = (1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch)))
        explained_var_new = (1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch)))
        # entropy 信息熵，越小越好
        print(("TRAIN kl:{:.5f},lr_multiplier:{:.3f},v_loss:{:.5f},p_loss:{:.5f},entropy:{:.5f},var_old:{:.5f},var_new:{:.5f}"
                      ).format(kl, self.lr_multiplier, v_loss, p_loss, entropy, explained_var_old, explained_var_new))
        return loss, entropy  

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

