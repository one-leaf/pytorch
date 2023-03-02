import os, glob, pickle

from model import PolicyValueNet, data_dir, data_wait_dir, model_file
from agent import Agent, ACTIONS
from mcts_single import MCTSPlayer

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
        if "piececount" not in result:
            result["piececount"]=[]
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
            if mcts_prob[0]<0.1:
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

        if os.path.exists(model_file):
            if time.time()-os.path.getmtime(model_file)>30*60:
                print("超过30分钟模型都没有更新了，停止训练")
                time.sleep(60)
                return

        # 游戏代理
        borads = []

        # 开始游戏
        policy_value_net = PolicyValueNet(GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=model_file)
        bestmodelfile = model_file+"_best"

        
        game_json = os.path.join(data_dir, "result.json")
        result = self.read_status_file(game_json)     
  
        data = {"steps":[],"shapes":[],"last_state":0,"score":0,"piece_count":0}
        start_time = time.time()
        print('start game time:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # 先运行测试
        his_pieces = []
        for _ in range(5):
            agent = Agent(isRandomNextPiece=True,)
            agent.show_mcts_process= True
            agent.id = 0

            for i in count():
                move_probs, state_value = policy_value_net.policy_value_fn(agent)
                action, acc_ps = 0, 0
                for a, p in move_probs:
                    if p > acc_ps:
                        action, acc_ps = a, p
                _, reward = agent.step(action)
                if reward > 0:
                    print("#"*40, 'score:', agent.score, 'height:', agent.pieceheight, 'piece:', agent.piececount, "shape:", agent.fallpiece["shape"], \
                        'step:', agent.steps, "step time:", round((time.time()-start_time)/i,3),'avg_score:', result["total"]["avg_score"])            

                if agent.terminal:            
                    result["total"]["avg_score"] = result["total"]["avg_score"]*0.99 + agent.score*0.01
                    result["lastupdate"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    break
            agent.print()

            # 判断是否需要重新玩,如果当前奖励小于平均奖励的1/2，放到运行池训练
            if agent.score < result["total"]["avg_score"]/2:
                his_pieces = agent.tetromino.piecehis
                filename = "{}-{}.pkl".format(agent.score, int(round(time.time() * 1000000)))
                savefile = os.path.join(self.waitplaydir, filename)
                with open(savefile, "wb") as fn:
                    pickle.dump(his_pieces, fn)

        self.save_status_file(result, game_json) 

        # 正式运行
        player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)

        files = glob.glob(os.path.join(self.waitplaydir, "*.pkl"))
        if len(files)>0:
            his_pieces_file = random.choice(files)
            with open(his_pieces_file,"rb") as fn:
                his_pieces = pickle.load(fn)     
            print("replay test again, load file:", his_pieces_file)
            print([p["shape"] for p in his_pieces])
            print("delete", his_pieces_file)
            os.remove(his_pieces_file)
            agent = Agent(isRandomNextPiece=True, nextpieces=his_pieces)
        else:
            agent = Agent(isRandomNextPiece=True, )

        agent.show_mcts_process= True
        agent.id = 0

        for i in count():
            _step={"step":i, "curr_player":agent.id}
            _step["state"] = agent.current_state()    
            # print(_step["state"])           
            _step["piece_count"] = agent.piececount               
            _step["shape"] = agent.fallpiece["shape"]
            _step["pre_piece_height"] = agent.pieceheight

            action, move_probs, state_value, qval, acc_ps, depth, ns = player.get_action(agent, agent.id, temp=1, avg_ns=result["total"]["ns"], avg_piececount=result["total"]["piececount"]) 

            _, reward = agent.step(action)

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

            # 这里的奖励是消除的行数
            if reward > 0:
                repeat_count = 40
                # print(_step["state"][0])
                # print(_step["state"][-1])
                print("#"*repeat_count, 'score:', agent.score, 'height:', agent.pieceheight, 'piece:', agent.piececount, "shape:", agent.fallpiece["shape"], \
                    'step:', agent.steps, "step time:", round((time.time()-start_time)/i,3),'player:', agent.id)
                # if agent.score>result["total"]["reward"]+20: game_stop=True

            # 如果训练次数超过了最大次数，则直接终止训练

            if agent.terminal:
                data["score"] = agent.score
                data["piece_count"] = agent.piececount
                data["piece_height"] = agent.pieceheight
                borads.append(agent.board)                    

                game_reward =  agent.score 
                result = self.read_status_file(game_json)

                paytime = time.time()-start_time
                steptime = paytime/agent.steps

                result["total"]["avg_score_ex"] = result["total"]["avg_score_ex"]*0.999 + agent.score*0.001 
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
                    result["total"]["step_time"] = result["total"]["step_time"]*0.99 + steptime*0.01
            
                if game_reward>result["best"]["reward"]:
                    result["best"]["reward"] = game_reward
                    result["best"]["agent"] = result["total"]["agent"]
                else:
                    result["best"]["reward"] = round(result["best"]["reward"] - 0.9999,4)

                if result["total"]["reward"]==0:
                    result["total"]["reward"] = game_reward
                else:
                    result["total"]["reward"] = result["total"]["reward"]*0.99 + game_reward*0.01

                if result["total"]["piececount"]==0:
                    result["total"]["piececount"] = agent.piececount
                else:
                    result["total"]["piececount"] = result["total"]["piececount"]*0.99 + agent.piececount*0.01

                # 计算 acc 看有没有收敛
                pacc = []
                vacc = []
                depth = []
                ns = []
                winner  = True if agent.piececount > result["total"]["piececount"] else False
                for step in data["steps"]:
                    pacc.append(step["acc_ps"])
                    depth.append(step["depth"])
                    ns.append(step["ns"])
                    if (not winner and step["state_value"]>0) or (winner and step["state_value"]<0):
                        vacc.append(0)
                    else:
                        vacc.append(1)

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
                    result["reward"].append(round(result["total"]["avg_score"],1))
                    result["depth"].append(round(result["total"]["depth"],1))
                    result["pacc"].append(round(result["total"]["pacc"],3))
                    result["vacc"].append(round(result["total"]["vacc"],3))
                    result["time"].append(round(result["total"]["step_time"],1))
                    result["ns"].append(round(result["total"]["ns"],1))
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

        winner = True if agent.piececount > result["total"]["piececount"] else False

        print("winner: %s piececount: %s %s" %(winner, agent.piececount, result["total"]["piececount"]))

        # 更新reward和score，reward为胜负，[1|-1|0]；score 为本步骤以后一共消除的行数
        step_count = len(data["steps"])
       
        # 奖励的分配
        # 重新定义价值为探索深度，因此value应该是[0,-X]
        # 如何评价最后的得分？最完美的情况（填满比）

        # 整体的价值
        pieces_value = [0 for _ in range(agent.piececount)]
        # 局部的收益
        pieces_score = [0 for _ in range(agent.piececount)]
        # 奖励的位置
        pieces_reward = [0 for _ in range(agent.piececount)]
        # 奖励的位置
        pieces_steps = [0 for _ in range(agent.piececount)]

        # 游戏的最终得分（0~-1）        
        r = agent.get_final_reward()
        # 每个方块的价值
        _r = agent.get_singe_piece_value()
        for m in range(agent.piececount):
            pieces_value[m] = r  - _r*(agent.piececount-m-1)

        # 统计所有获得奖励的方块
        for m in range(step_count):
            pieces_steps[data["steps"][m]["piece_count"]] = m
            if data["steps"][m]["reward"]>0:
                pieces_reward[data["steps"][m]["piece_count"]] = 1

        # 统计局部的收益
        for m in range(agent.piececount):                    
            pieces_score[m] = data["steps"][pieces_steps[m]]["pre_piece_height"] + 0.4 - data["steps"][pieces_steps[m]]["piece_height"]

        print()
        print(i, pieces_reward)
        print()
        print(i, pieces_value)
        print()
        print(i, pieces_score)
        print()

        # 分配收益到每一步
        for m in range(step_count):
            p_id = data["steps"][m]["piece_count"]
            s_step = -1 if p_id == 0 else pieces_steps[p_id-1]
            e_step = pieces_steps[p_id]
            s_value = 0 if p_id == 0 else pieces_value[p_id-1]
            e_value = pieces_value[p_id]
            # 局部收益始终初始为 0
            s_score = 0 # 0 if p_id == 0 else pieces_score[p_id-1] 
            e_score = pieces_score[p_id]
            if s_step==e_step:
                data["steps"][m]["value"]=e_value
                data["steps"][m]["score"]=e_score
            else:        
                data["steps"][m]["value"]=s_value+(m-s_step)/(e_step-s_step)*(e_value-s_value)
                data["steps"][m]["score"]=s_score+(m-s_step)/(e_step-s_step)*(e_score-s_score)

        # # 分配收益到整个方块
        # for m in range(step_count):
        #     p_id = data["steps"][m]["piece_count"]
        #     data["steps"][m]["value"]=pieces_value[p_id]
        #     data["steps"][m]["score"]=pieces_score[p_id]

        print(i,"score:",data["score"],"piece_count:",data["piece_count"],"piece_height:",data["piece_height"],"steps:",step_count)
       
        states, mcts_probs, values, score= [], [], [], []

        for step in data["steps"]:
            states.append(step["state"])
            mcts_probs.append(step["move_probs"])
            values.append(step["value"])
            score.append(step["score"])

        assert len(states)>0
        assert len(states)==len(values)
        assert len(states)==len(mcts_probs)
        assert len(states)==len(score)

        print("TRAIN Self Play end. length: %s value sum: %s saving ..." % (len(states),sum(values)))

        # 保存对抗数据到data_buffer
        filetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # 现在第一层改为了横向，所以不能做图片左右翻转增强
        # for i, obj in enumerate(self.get_equi_data(states, mcts_probs, values, score)):
        for i, obj in enumerate(zip(states, mcts_probs, values, score)):
            filename = "{}-{}.pkl".format(filetime, i)
            savefile = os.path.join(data_wait_dir, filename)
            with open(savefile, "wb") as fn:
                pickle.dump(obj, fn)
        print("saved file basename:", filetime, "length:", i+1)

        # 删除训练集
        if agent.piececount < result["total"]["piececount"]/2:
            filename = "{}-{}.pkl".format(agent.score, int(round(time.time() * 1000000)))
            his_pieces_file = os.path.join(self.waitplaydir, filename)
            with open(his_pieces_file, "wb") as fn:
                pickle.dump(agent.tetromino.piecehis, fn)

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

