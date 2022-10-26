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
        self.n_playout = 128  # 每个动作的模拟战记录个数，影响后续 128/2 = 66；64/16 = 4个方块 的走法
        self.play_batch_size = 5 # 每次自学习次数
        self.buffer_size = 1000000  # cache对次数
        self.epochs = 2  # 每次更新策略价值网络的训练步骤数, 推荐是5
        self.kl_targ = 0.02  # 策略价值网络KL值目标
        self.best_win_ratio = 0.0
        
        # aplhazero 的最佳值是 4 
        # MCTS child权重， 用来调节MCTS搜索深度，越大搜索越深，越相信概率，越小越相信Q 的程度 默认 5
        # 由于value完全用结果胜负来拟合，所以value不稳，只能靠概率p拟合，最后带动value来拟合
        self.c_puct = 5  


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
            result={"reward":[], "depth":[], "pacc":[], "vacc":[], "time":[], "ns":[]}
        if "total" not in result:
            result["total"]={"agent":0, "pacc":0, "vacc":0, "ns":0, "reward":0, "depth":0, "step_time":0, "_agent":0}
        if "best" not in result:
            result["best"]={"reward":0, "agent":0}

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
        agent = Agent(isRandomNextPiece=False)

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
        agent.id = 0
        agent.id = 1
        
        game_json = os.path.join(data_dir, "result.json")
        result = self.read_status_file(game_json)
        
        # 由于动态cpuct并没有得到一个好的结果，所以关闭
        # 读取各自的动态cpuct
        # cpuct_result = game_result["cpuct"]
        # cpuct_list = sorted(cpuct_result, key=lambda x : cpuct_result[x]["count"])
        # cpuct = float(cpuct_list[0])
        # print("cpuct1:", cpuct_result, "-->", cpuct_list, "cpuct1:", cpuct, "n_playout:", self.n_playout)
        # cpuct_list.sort() 

        if os.path.exists(bestmodelfile):
            policy_value_net_best = PolicyValueNet(GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=bestmodelfile)
            # if random.random()>0.5:
            #     player = MCTSPlayer((policy_value_net.policy_value_fn, policy_value_net_best.policy_value_fn), c_puct=self.c_puct, n_playout=self.n_playout)
            # else:
            #     player = MCTSPlayer((policy_value_net_best.policy_value_fn, policy_value_net.policy_value_fn), c_puct=self.c_puct, n_playout=self.n_playout)
            # player = MCTSPlayer(policy_value_net_best.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)
            c = random.choice([0,1,2,3])
            if c == 0:
                player = MCTSPlayer(policy_value_net_best.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)
            elif c == 1:
                player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)    
            elif c == 2:
                player = MCTSPlayer((policy_value_net.policy_value_fn, policy_value_net_best.policy_value_fn), c_puct=self.c_puct, n_playout=self.n_playout)    
            elif c == 3:
                player = MCTSPlayer((policy_value_net_best.policy_value_fn, policy_value_net.policy_value_fn), c_puct=self.c_puct, n_playout=self.n_playout)    
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
            # print(_step["state"])           
            _step["piece_count"] = game.piececount               
            _step["shape"] = game.fallpiece["shape"]
            _step["pre_piece_height"] = game.pieceheight

            action, move_probs, state_value, qval, acc_ps, depth, ns = player.get_action(games, curr_player, temp=result["total"]["ns"]) 
            _, reward = game.step(action)

            _step["piece_height"] = game.pieceheight
            _step["reward"] = reward if reward>0 else 0
            _step["move_probs"] = move_probs
            _step["state_value"] = state_value
            _step["qval"] = qval
            _step["acc_ps"] = acc_ps
            _step["depth"] = depth
            _step["ns"] = ns

            data["steps"].append(_step)

            # 这里的奖励是消除的行数
            if reward > 0:
                repeat_count = 40
                print(_step["state"])
                print("#"*repeat_count, 'score:', game.score, 'height:', game.pieceheight, 'piece:', game.piececount, "shape:", game.fallpiece["shape"], \
                    'step:', game.steps, "step time:", round((time.time()-start_time)/i,3),'player:', curr_player)

            # 如果训练次数超过了最大次数，则直接终止训练
            if i >= 10000: game_stop=True

            if game.pieceheight>=5:
                game_stop=True
            if abs(games[0].pieceheight-games[1].pieceheight)>=2:
                game_stop=True
                
            # if games[0].piececount>1 and games[1].piececount>1 and games[0].piececount==games[1].piececount:
            #     if (games[0].score != games[1].score and games[0].pieceheight!=games[1].pieceheight): 
            #         game_stop=True

            if game.terminal or game_stop:
                for _game, _data in zip(games, game_datas):
                    _data["score"] = _game.score
                    _data["piece_count"] = _game.piececount
                    _data["piece_height"] = _game.pieceheight
                    borads.append(_game.board)                    

                game_reward =  sum([_game.score for _game in games])/len(games)
                result = self.read_status_file(game_json)

                paytime = time.time()-start_time
                steptime = paytime/sum([_game.steps for _game in games])

                result["total"]["agent"] += 2
                result["total"]["_agent"] += 2

                if result["total"]["step_time"]==0:
                    result["total"]["step_time"] = steptime
                else:
                    result["total"]["step_time"] = result["total"]["step_time"]*0.99 + steptime*0.01
            
                if game_reward>result["best"]["reward"]:
                    result["best"]["reward"] = game_reward
                    result["best"]["agent"] = result["total"]["agent"]

                if result["total"]["reward"]==0:
                    result["total"]["reward"] = game_reward
                else:
                    result["total"]["reward"] = result["total"]["reward"]*0.99 + game_reward*0.01

                # 计算 acc 看有没有收敛

                pacc = []
                vacc = []
                depth = []
                ns = []
                winner  = 0 if games[0].pieceheight < games[1].pieceheight else 1
                for j, _game, _data in zip([0,1], games, game_datas):
                    for step in _data["steps"]:
                        pacc.append(step["acc_ps"])
                        depth.append(step["depth"])
                        ns.append(step["ns"])
                        if (j==winner and step["state_value"]>0) or (j!=winner and step["state_value"]<0):
                            vacc.append(1)
                        else:
                            vacc.append(0)

                pacc = np.average(pacc)
                vacc = np.average(vacc)
                depth = np.average(depth)
                ns = np.average(ns)

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
                    result["reward"].append(round(result["total"]["reward"],1))
                    result["depth"].append(round(result["total"]["depth"],1))
                    result["pacc"].append(round(result["total"]["pacc"],3))
                    result["vacc"].append(round(result["total"]["vacc"],3))
                    result["time"].append(round(result["total"]["step_time"],1))
                    result["ns"].append(round(result["total"]["ns"],1))
                    result["total"]["_agent"] -= 50 

                    while len(result["reward"])>200:
                        result["reward"].remove(result["reward"][0])
                    while len(result["depth"])>200:
                        result["depth"].remove(result["depth"][0])
                    while len(result["pacc"])>200:
                        result["pacc"].remove(result["pacc"][0])
                    while len(result["vacc"])>200:
                        result["vacc"].remove(result["vacc"][0])
                    while len(result["time"])>200:
                        result["time"].remove(result["time"][0])
                    while len(result["ns"])>200:
                        result["ns"].remove(result["ns"][0])
                   
                    if max(result["depth"])==result["depth"][-1]:
                        newmodelfile = model_file+"_depth_"+str(result["depth"][-1])
                        if not os.path.exists(newmodelfile):
                            policy_value_net.save_model(newmodelfile)
                       
                    newmodelfile = model_file+"_"+str(result["total"]["agent"])
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

        if games[0].score == 0 and games[1].score == 0:
            winner = -1
        else:
            winner = 1 if games[0].pieceheight > games[1].pieceheight else 0

        # 更新reward和score，reward为胜负，[1|-1|0]；score 为本步骤以后一共消除的行数
        for i, data in enumerate(game_datas):
            step_count = len(data["steps"])
            piece_count = -1
            v = 1 if i==winner else -1 
            score = 0
            vacclist=[]
            slist=[]
            pacclist=[]
            vacc_sum = 0
            s_sum = 0
            pacc_sum = 0
            d_sum = 0
            for j in range(step_count-1,-1,-1):
                if piece_count!=data["steps"][j]["piece_count"]:
                    piece_count = data["steps"][j]["piece_count"]
                    score += data["steps"][j]["reward"]
                    vacclist.insert(0, round(data["steps"][j]["state_value"],2))
                    slist.insert(0, score)
                    pacclist.insert(0, round(max(data["steps"][j]["move_probs"]),2))
                q = data["steps"][j]["qval"] 
                data["steps"][j]["reward"] = v
                data["steps"][j]["score"] = q

                vacc_sum += abs(v-data["steps"][j]["state_value"])
                s_sum += score
                pacc_sum += abs(1-max(data["steps"][j]["move_probs"]))
                d_sum += data["steps"][j]["depth"]
            print(i,"value:",v,"score:",data["score"],"piece_count:",data["piece_count"],"piece_height:",data["piece_height"],"steps:",step_count,"depth:",d_sum/step_count)
            print(i,"avg_score:",s_sum/step_count, slist)
            print(i,"v_acc:",vacc_sum/step_count, vacclist)
            print(i,"p_acc:",pacc_sum/step_count, pacclist)
       
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
        # 现在第一层改为了横向，所以不能做图片左右翻转增强
        # for i, obj in enumerate(self.get_equi_data(states, mcts_probs, values, qval)):
        for i, obj in enumerate(zip(states, mcts_probs, values, qval)):
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

