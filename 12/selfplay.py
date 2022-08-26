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
        self.n_playout = 128  # 每个动作的模拟战记录个数，不能用32，因为收敛太慢了
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
            result["cpuct"]={"0.5":{"count":0,"value":0},"0.6":{"count":0,"value":0}}    
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

        # 游戏代理
        agent = Agent(max_height=5)

        borads = []
        game_datas = []

        # 开始游戏
        policy_value_net = PolicyValueNet(GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=model_file)
        # agent.show_mcts_process= True
        # 同时开两个游戏
        game1 = copy.deepcopy(agent)
        game2 = copy.deepcopy(agent)
        game1_json = os.path.join(data_dir, "result.json")
        game2_json = os.path.join(data_dir, "result_flip_v.json")
        game1_result = self.read_status_file(game1_json)
        game2_result = self.read_status_file(game2_json)
        
        # 读取各自的动态cpuct
        cpuct1_result = game1_result["cpuct"]
        cpuct1_list = sorted(cpuct1_result, key=lambda x : cpuct1_result[x]["count"])
        cpuct1 = float(cpuct1_list[0])
        print("cpuct1:", cpuct1_result, "-->", cpuct1_list, "cpuct1:", cpuct1, "n_playout:", self.n_playout)
        cpuct1_list.sort()

        cpuct2_result = game2_result["cpuct"]
        cpuct2_list = sorted(cpuct2_result, key=lambda x : cpuct2_result[x]["count"])
        cpuct2 = float(cpuct2_list[0])
        print("cpuct2:",cpuct2_result, "-->", cpuct2_list, "cpuct2:", cpuct2, "n_playout:", self.n_playout)
        cpuct2_list.sort()
        player1 = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=cpuct1, n_playout=self.n_playout, player_id=0)
        player2 = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=cpuct2, n_playout=self.n_playout, player_id=1)

        data1 = {"steps":[],"shapes":[],"last_state":0,"score":0,"piece_count":0}
        data2 = {"steps":[],"shapes":[],"last_state":0,"score":0,"piece_count":0}

        start_time = time.time()
        print('start game time:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        game_stop= False
        for i in count():

            # 每个都走一步
            for game, player, data, jsonfile, cpuct, cpuct_list in [(game1,player1,data1,game1_json,cpuct1,cpuct1_list), (game2,player2,data2,game2_json,cpuct2,cpuct2_list)]:
                _step={"step":i}
                _step["state"] = game.current_state()               
                _step["piece_count"] = game.piececount               
                _step["shape"] = game.fallpiece["shape"]
                _step["pre_piece_height"] = game.pieceheight

                action, move_probs, state_value = player.get_action(game, temp=1) 
                _, reward = game.step(action)

                _step["piece_height"] = game.pieceheight
                _step["reward"] = reward if reward>0 else 0
                _step["move_probs"] = move_probs
                _step["state_value"] = state_value

                data["steps"].append(_step)

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

                    self.save_status_file(result, jsonfile)
                    repeat_count = 20 if game==game2 else 40
                    print("#"*repeat_count, 'score:', game.score, 'height:', game.pieceheight, 'piece:', game.piececount, "shape:", game.fallpiece["shape"], \
                        'step:', game.steps, "step time:", round((time.time()-start_time)/(i*2.),3),'player:', player.player_id)


            if game1.terminal or game2.terminal or game_stop:
                for game, player, data, jsonfile, cpuct, cpuct_list in [(game1,player1,data1,game1_json,cpuct1,cpuct1_list), (game2,player2,data2,game2_json,cpuct2,cpuct2_list)]:

                    data["score"] = game.score
                    data["piece_count"] = game.piececount
                    data["piece_height"] = game.pieceheight

                    game_datas.append(data)
                    borads.append(game.board)

                    # 更新状态
                    game_reward =  game.score   

                    result = self.read_status_file(jsonfile)

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
                        result["best"]["agent"] = result["agent"]+1

                    result["agent"] += 1
                    result["curr"]["reward"] += game.score
                    result["curr"]["pieces"] += game.piececount
                    result["curr"]["agent1000"] += 1
                    result["curr"]["agent100"] += 1

                    if not game.terminal:
                        if "win" in result["curr"]:
                            result["curr"]["win"] += 1
                        else:
                            result["curr"]["win"] = 1

                    # 计算 acc 看有没有收敛
                    v = -1
                    if not game.terminal:
                        v = 1
                    elif game1.terminal and game2.terminal:
                        v = 0
                    acc = 0
                    for step in data["steps"]:
                        acc += (step["state_value"]-v)**2
                    acc = acc/len(data["steps"])

                    if result["QVal"]==0:
                        result["QVal"] = acc
                    else:
                        result["QVal"] = round(result["QVal"]*0.99 + acc*0.01,2)   

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

                        # 每50局更新一次cpuct参数
                        count0=result["cpuct"][cpuct_list[0]]["count"]
                        count1=result["cpuct"][cpuct_list[1]]["count"]
                        if count0>10 and count1>10:
                            v0 = result["cpuct"][cpuct_list[0]]["value"]/count0
                            v1 = result["cpuct"][cpuct_list[1]]["value"]/count1
                            if v0 > v1:
                                cpuct = round(float(cpuct_list[0])-0.1,1)
                                if cpuct<=0.1:
                                    result["cpuct"] = {"0.1":{"count":0,"value":0}, "0.2":{"count":0,"value":0}}
                                else:
                                    result["cpuct"] = {str(cpuct):{"count":0,"value":0}, str(round(cpuct+0.1,1)):{"count":0,"value":0}}
                            else:
                                cpuct = round(float(cpuct_list[0])+0.1,1)
                                result["cpuct"] = {str(cpuct):{"count":0,"value":0}, str(round(cpuct+0.1,1)):{"count":0,"value":0}}

                        if max(result["reward"])==result["reward"][-1]:
                            newmodelfile = model_file+"_reward_"+str(result["reward"][-1])
                            if not os.path.exists(newmodelfile):
                                policy_value_net.save_model(newmodelfile)

                    if result["curr"]["agent1000"]>500:
                        result["curr"]={"reward":0,"pieces":0,"agent1000":0,"agent100":0,"height":0}

                        newmodelfile = model_file+"_"+str(result["agent"])
                        if not os.path.exists(newmodelfile):
                            policy_value_net.save_model(newmodelfile)
                    result["lastupdate"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    self.save_status_file(result, jsonfile) 

                    # game.print()
                    # print('reward:', game.score, "Qval:", game_reward, 'len:', i, "piececount:", game.piececount, "time:", time.time()-start_time)
                    # print("pay:", time.time() - start_time , "s\n" )

                break

            # 如果训练次数超过了最大次数，则直接终止训练
            if i >= 10000: game_stop=True

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

        ## 放弃 按0.99的衰减更新reward
        # 只关注最后一次得分方块的所有步骤,将消行方块的所有步骤的得分都设置为1
        winner = [0, 0] 
        if game1.terminal and not game2.terminal:
            winner[0] = -1
            winner[1] = 1
        elif not game1.terminal and game2.terminal:
            winner[0] = 1
            winner[1] = -1

        # 更新reward和score，reward为胜负，[1|-1|0]；score 为本步骤以后一共消除的行数
        for i, data in enumerate(game_datas):
            step_count = len(data["steps"])
            piece_count = -1
            v = winner[i]
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
                    vlist.insert(0,v)
                    slist.insert(0, score)
                    acclist.insert(0, data["steps"][j]["state_value"])
                data["steps"][j]["reward"] = v
                data["steps"][j]["score"] = score
                v_sum += v
                s_sum += score
                acc_sum += (data["steps"][j]["state_value"]-v)**2
            print("score","max height:",data["piece_height"],"avg:",s_sum/step_count, slist)
            print("value","piece len:",len(vlist),"avg:",v_sum/step_count, vlist)
            print("acc","steps len:",step_count,"avg:",acc_sum/step_count, acclist)
       
        states, mcts_probs, values, score= [], [], [], []

        for data in game_datas:
            for step in data["steps"]:
                states.append(step["state"])
                mcts_probs.append(step["move_probs"])
                values.append(step["reward"])
                score.append(step["score"])

        assert len(states)>0
        assert len(states)==len(values)
        assert len(states)==len(mcts_probs)
        assert len(states)==len(score)

        print("TRAIN Self Play end. length:%s value sum:%s saving ..." % (len(states),sum(values)))

        # 保存对抗数据到data_buffer
        filetime = datetime.datetime.now().isoformat()
        print("save file basename:", filetime)
        for i, obj in enumerate(self.get_equi_data(states, mcts_probs, values, score)):
        # for i, obj in enumerate(zip(states, mcts_probs, values, score)):
            filename = "{}-{}.pkl".format(filetime, i)
            savefile = os.path.join(data_wait_dir, filename)
            with open(savefile, "wb") as fn:
                pickle.dump(obj, fn)

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

