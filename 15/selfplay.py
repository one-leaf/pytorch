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
        # self.n_playout = 512  # 每个动作的模拟战记录个数，影响后续 512/2 = 256；256/16 = 16个方块 的走法
        # 64/128 都不行
        self.n_playout = 256  # 每个动作的模拟战记录个数，影响后续 128/2 = 66；64/16 = 4个方块 的走法
        self.play_batch_size = 5 # 每次自学习次数
        self.buffer_size = 1000000  # cache对次数
        self.epochs = 2  # 每次更新策略价值网络的训练步骤数, 推荐是5
        self.kl_targ = 0.02  # 策略价值网络KL值目标
        self.best_win_ratio = 0.0
        
        # aplhazero 的最佳值是 4 
        self.c_puct = 3  # MCTS child权重， 用来调节MCTS搜索深度，越大搜索越深，越相信概率，越小越相信Q 的程度 默认 5


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
            result={"agent":0, "reward":[], "steps":[], "accs":[], "acc":0}
        if "curr" not in result:
            result["curr"]={"reward":0, "step":0, "agent500":0, "agent50":0}
        if "best" not in result:
            result["best"]={"reward":0, "agent":0}
        if "cpuct" not in result:
            result["cpuct"]={"0.5":{"count":0,"value":0},"0.6":{"count":0,"value":0}}    
        for key in result["cpuct"]:
            if not isinstance(result["cpuct"][key], dict):
                result["cpuct"][key] = {"count":0,"value":result["cpuct"][key]}
        if "time" not in result:
            result["time"]={"agent_time":0,"step_time":0,"step_times":[]}
        if "vars" not in result or "avg" not in result["vars"]:
            # 缺省std = sqrt(2)
            result["vars"]={"max":1, "min":-1, "std":1, "avg":0}

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
            # 如果旋转的概率不大，则可以反转左右来增加样本数
            if mcts_prob[1]<0.1:
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

        # 游戏代理
        agent = Agent(max_height=10, isRandomNextPiece=False)

        borads = []

        # 开始游戏
        policy_value_net = PolicyValueNet(GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=model_file)
        bestmodelfile = model_file+"_best"

        # 同时开两个游戏
        # if random.random()>0.9:
        #     agent2 = copy.deepcopy(agent)
        # else:
        # agent2 = Agent(max_height=10, isRandomNextPiece=True)
        agent2 = copy.deepcopy(agent)
        games = (agent, agent2)

        agent.show_mcts_process= True
        agent2.show_mcts_process= True


        game_json = os.path.join(data_dir, "result.json")
        # game_result = self.read_status_file(game_json)
        
        # 由于动态cpuct并没有得到一个好的结果，所以关闭
        # 读取各自的动态cpuct
        # cpuct_result = game_result["cpuct"]
        # cpuct_list = sorted(cpuct_result, key=lambda x : cpuct_result[x]["count"])
        # cpuct = float(cpuct_list[0])
        # print("cpuct1:", cpuct_result, "-->", cpuct_list, "cpuct1:", cpuct, "n_playout:", self.n_playout)
        # cpuct_list.sort() 

        if os.path.exists(bestmodelfile):
            policy_value_net_best = PolicyValueNet(GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=bestmodelfile)
            if random.random()>0.5:
                player = MCTSPlayer((policy_value_net.policy_value_fn, policy_value_net_best.policy_value_fn), c_puct=self.c_puct, n_playout=self.n_playout)
            else:
                player = MCTSPlayer((policy_value_net_best.policy_value_fn, policy_value_net.policy_value_fn), c_puct=self.c_puct, n_playout=self.n_playout)
            # player = MCTSPlayer(policy_value_net_best.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)
            # c = random.choice([0,1,2,3])
            # if c == 0:
            #     player = MCTSPlayer(policy_value_net_best.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)
            # elif c == 1:
            #     player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)    
            # elif c == 2:
            #     player = MCTSPlayer((policy_value_net.policy_value_fn, policy_value_net_best.policy_value_fn), c_puct=self.c_puct, n_playout=self.n_playout)    
            # elif c == 3:
            #     player = MCTSPlayer((policy_value_net_best.policy_value_fn, policy_value_net.policy_value_fn), c_puct=self.c_puct, n_playout=self.n_playout)    
        else:
            player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)

        # if random.random()>0.5 and os.path.exists(bestmodelfile):
        #     policy_value_net_best = PolicyValueNet(GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=bestmodelfile)
        #     player = MCTSPlayer(policy_value_net_best.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)
        # else:
        #     player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)
    
        data0 = {"steps":[],"shapes":[],"last_state":0,"score":0,"piece_count":0}
        data1 = {"steps":[],"shapes":[],"last_state":0,"score":0,"piece_count":0}
        game_datas = (data0, data1)

        start_time = time.time()
        print('start game time:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        game_stop= False
        for i in count():
            curr_player = i%2
            game = games[curr_player]
            data = game_datas[curr_player]            

            _step={"step":i, "curr_player":curr_player}
            _step["state"] = game.current_state()               
            _step["piece_count"] = game.piececount               
            _step["shape"] = game.fallpiece["shape"]
            _step["pre_piece_height"] = game.pieceheight

            # action, move_probs, state_value, qval = player.get_action(games, curr_player, temp=1/(1+game.pieceheight)) 
            action, move_probs, state_value, qval = player.get_action(games, curr_player, temp=1) 
            _, reward = game.step(action)

            _step["piece_height"] = game.pieceheight
            _step["reward"] = reward if reward>0 else 0
            _step["move_probs"] = move_probs
            _step["state_value"] = state_value
            _step["qval"] = qval

            data["steps"].append(_step)

            # 这里的奖励是消除的行数
            if reward > 0:
                repeat_count = 40
                print(_step["state"])
                print("#"*repeat_count, 'score:', game.score, 'height:', game.pieceheight, 'piece:', game.piececount, "shape:", game.fallpiece["shape"], \
                    'step:', game.steps, "step time:", round((time.time()-start_time)/i,3),'player:', curr_player)

            # 如果训练次数超过了最大次数，则直接终止训练
            if i >= 10000: game_stop=True
            if abs(games[0].pieceheight-games[1].pieceheight)>2 and games[0].piececount>0 and games[1].piececount>0 and games[0].piececount==games[1].piececount: 
                game_stop=True

            if game.terminal or game_stop:
                for _game, _data in zip(games, game_datas):
                    _data["score"] = _game.score
                    _data["piece_count"] = _game.piececount
                    _data["piece_height"] = _game.pieceheight
                    borads.append(_game.board)                    

                game_reward =  sum([_game.score for _game in games])
                game_step =  sum([_game.steps for _game in games])

                result = self.read_status_file(game_json)

                paytime = time.time()-start_time
                steptime = paytime/sum([_game.steps for _game in games])

                if result["time"]["agent_time"]==0:
                    result["time"]["agent_time"] = paytime
                    result["time"]["step_time"] = steptime
                else:
                    result["time"]["agent_time"] = round(result["time"]["agent_time"]*0.99+paytime*0.01, 3)
                    d = game.steps/10000.0
                    if d>1 : d = 0.99
                    result["time"]["step_time"] = round(result["time"]["step_time"]*(1-d)+steptime*d, 3)
            
                # 记录当前cpuct的统计结果
                # cpuct_str = str(cpuct)
                # if cpuct_str in result["cpuct"]:
                #     result["cpuct"][cpuct_str]["value"] = result["cpuct"][cpuct_str]["value"]+game_step
                #     result["cpuct"][cpuct_str]["count"] = result["cpuct"][cpuct_str]["count"]+1         

                if game_reward>result["best"]["reward"]:
                    result["best"]["reward"] = game_reward
                    result["best"]["score"] = game.score
                    result["best"]["agent"] = result["agent"]

                if not game_stop:
                    result["agent"] += 2
                    result["curr"]["reward"] += game_reward
                    result["curr"]["step"] += game_step
                    result["curr"]["agent500"] += 2
                    result["curr"]["agent50"] += 2

                # 计算 acc 看有没有收敛

                acc = []
                for _game, _data in zip(games, game_datas):
                    for step in _data["steps"]:
                        acc.append(abs((step["state_value"]-step["qval"])))
                acc = np.average(acc)

                if result["acc"]==0:
                    result["acc"] = round(acc,2)
                else:
                    result["acc"] = round(result["acc"]*0.99 + acc*0.01,2)   

                if result["curr"]["agent50"]>50:
                    result["reward"].append(round(result["curr"]["reward"]/result["curr"]["agent500"],2))
                    result["steps"].append(round(result["curr"]["step"]/result["curr"]["agent500"],2))
                    result["accs"].append(round(result["acc"],2))
                    result["time"]["step_times"].append(result["time"]["step_time"])
                    result["curr"]["agent50"] -= 50 
                    while len(result["reward"])>200:
                        result["reward"].remove(result["reward"][0])
                    while len(result["accs"])>200:
                        result["accs"].remove(result["accs"][0])
                    while len(result["time"]["step_times"])>200:    
                        result["time"]["step_times"].remove(result["time"]["step_times"][0])

                    # 每50局更新一次cpuct参数
                    # count0=result["cpuct"][cpuct_list[0]]["count"]
                    # count1=result["cpuct"][cpuct_list[1]]["count"]
                    # if count0>10 and count1>10:
                    #     v0 = result["cpuct"][cpuct_list[0]]["value"]/count0
                    #     v1 = result["cpuct"][cpuct_list[1]]["value"]/count1
                    #     if v0 > v1:
                    #         cpuct = round(float(cpuct_list[0])-0.1,1)
                    #         if cpuct<0.1:
                    #             result["cpuct"] = {"0.1":{"count":0,"value":0}, "0.2":{"count":0,"value":0}}
                    #         else:
                    #             result["cpuct"] = {str(cpuct):{"count":0,"value":0}, str(round(cpuct+0.1,1)):{"count":0,"value":0}}
                    #     else:
                    #         cpuct = round(float(cpuct_list[0])+0.1,1)
                    #         result["cpuct"] = {str(cpuct):{"count":0,"value":0}, str(round(cpuct+0.1,1)):{"count":0,"value":0}}

                    if max(result["steps"])==result["steps"][-1]:
                        newmodelfile = model_file+"_steps_"+str(result["steps"][-1])
                        if not os.path.exists(newmodelfile):
                            policy_value_net.save_model(newmodelfile)
                        
                if result["curr"]["agent500"]>500:
                    result["curr"]={"reward":0,"step":0,"agent500":0,"agent50":0}

                    newmodelfile = model_file+"_"+str(result["agent"])
                    if not os.path.exists(newmodelfile):
                        policy_value_net.save_model(newmodelfile)

                    lastmodelfile = model_file+"_last"                                        
                    if os.path.exists(bestmodelfile): os.remove(bestmodelfile)
                    if os.path.exists(lastmodelfile): os.rename(lastmodelfile, bestmodelfile)
                    policy_value_net.save_model(lastmodelfile)

                result["lastupdate"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.save_status_file(result, game_json) 

                break


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

        if games[0].terminal or games[1].terminal:
            winner = 1 if games[0].terminal else 0
        else:
            winner = 1 if games[0].pieceheight > games[1].pieceheight else 0

        # 更新reward和score，reward为胜负，[1|-1|0]；score 为本步骤以后一共消除的行数
        for i, data in enumerate(game_datas):
            step_count = len(data["steps"])
            piece_count = -1
            v = 1 if i==winner else -1 
            score = 0
            vlist=[]
            slist=[]
            acclist=[]
            v_sum = 0
            s_sum = 0
            acc_sum = 0
            for j in range(step_count-1,-1,-1):
                if piece_count!=data["steps"][j]["piece_count"]:
                    piece_count = data["steps"][j]["piece_count"]
                    score += data["steps"][j]["reward"]
                    vlist.insert(0,(v+data["steps"][j]["qval"])/2)
                    slist.insert(0, score)
                    acclist.insert(0, data["steps"][j]["state_value"])
                q = data["steps"][j]["qval"] 
                data["steps"][j]["reward"] = v
                data["steps"][j]["score"] = q
                v_sum += v
                s_sum += score
                acc_sum += abs(data["steps"][j]["state_value"]-data["steps"][j]["qval"])
            print("score","max height:",data["piece_height"],"avg:",s_sum/step_count, slist)
            print("qval","piece len:",len(vlist),"avg:",v_sum/step_count, vlist)
            print("acc","steps len:",step_count,"avg:",acc_sum/step_count, acclist)
       
        states, mcts_probs, values, qval= [], [], [], []

        for data in game_datas:
            for step in data["steps"]:
                states.append(step["state"])
                mcts_probs.append(step["move_probs"])
                values.append(step["reward"])
                qval.append(step["score"])

        assert len(states)>0
        assert len(states)==len(values)
        assert len(states)==len(mcts_probs)
        assert len(states)==len(qval)

        print("TRAIN Self Play end. length: %s value sum: %s saving ..." % (len(states),sum(values)))

        # 保存对抗数据到data_buffer
        filetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for i, obj in enumerate(self.get_equi_data(states, mcts_probs, values, qval)):
        # for i, obj in enumerate(zip(states, mcts_probs, values, qval)):
            filename = "{}-{}.pkl".format(filetime, i)
            savefile = os.path.join(data_wait_dir, filename)
            with open(savefile, "wb") as fn:
                pickle.dump(obj, fn)
        print("saved file basename:", filetime, "length:", i+1)

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

