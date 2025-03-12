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

from status import save_status_file, read_status_file, set_status_total_value

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
        self.n_playout = 64  # 每个动作的模拟战记录个数，影响后续 512/2 = 256；256/16 = 16个方块 的走法
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
        self.c_puct = 5  
        self.q_std = 1
        self.max_step_count = 10000 
        self.limit_steptime = 1  # 限制每一步的平均花费时间，单位秒，默认1秒

        # 等待训练的序列
        self.waitplaydir=os.path.join(data_dir,"replay")
        if not os.path.exists(self.waitplaydir): os.makedirs(self.waitplaydir)

        self.stop_mark_file = os.path.join(self.waitplaydir,"../stop")


    

    def get_equi_data(self, states, mcts_probs, values):
        """
        通过翻转增加数据集
        play_data: [(state, mcts_prob, values, score), ..., ...]
        """
        extend_data = []
        for i in range(len(states)):
            state, mcts_prob, value=states[i], mcts_probs[i], values[i]
            extend_data.append((state, mcts_prob, value))
            if mcts_prob[0]<0.2 and np.max(mcts_prob)>0.8: # 如果旋转的概率的不大，就做翻转
                equi_state = np.array([np.fliplr(s) for s in state])
                equi_mcts_prob = mcts_prob[[0,2,1,3,4]]
                extend_data.append((equi_state, equi_mcts_prob, value))
            # if i==0:
            #     print("state:",state)
            #     print("mcts_prob:",mcts_prob)
            #     print("equi_state:",equi_state)
            #     print("equi_mcts_prob:",equi_mcts_prob)
            #     print("value:",value)
        return extend_data

    def test_play(self,policy_value_net):
        # 先运行测试
        min_his_pieces = None
        min_his_pieces_len = 0
        min_pieces_count = -1
        min_removedlines = -1
        max_pieces_count = -1
        max_removedlines = -1

        for _ in range(self.play_size):
            agent = Agent(isRandomNextPiece=True)
            start_time = time.time()
            agent.show_mcts_process= False
            # agent.id = 0

            for i in range(self.max_step_count):
                action = policy_value_net.policy_value_fn_best_act(agent)
                _, reward = agent.step(action)
                if reward > 0:
                    print("#"*40, 'score:', agent.removedlines, 'height:', agent.pieceheight, 'piece:', agent.piececount, "shape:", agent.fallpiece["shape"], \
                        'step:', agent.steps, "step time:", round((time.time()-start_time)/i,3))            

                if agent.terminal: 
                    result = read_status_file()
                    set_status_total_value(result, "avg_score", agent.removedlines, 1/1000)
                    set_status_total_value(result, "avg_piececount", agent.piececount, 1/1000)
                    save_status_file(result)
                    result["lastupdate"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    break
                

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
                
            result = read_status_file()
            if agent.piececount<(result["total"]["avg_piececount"]+result["total"]["min_piececount"])/2:
                filename = "{}-{}.pkl".format("".join(min_his_pieces), min_his_pieces_len)
                his_pieces_file = os.path.join(self.waitplaydir, filename)
                print("save need replay", his_pieces_file)
                with open(his_pieces_file, "wb") as fn:
                    pickle.dump(min_his_pieces, fn)
                    
        result = read_status_file()
        
        set_status_total_value(result, "max_score", max_removedlines, 1/100)
        set_status_total_value(result, "max_piececount", max_pieces_count, 1/100)
        set_status_total_value(result, "min_score", min_removedlines, 1/100)
        set_status_total_value(result, "min_piececount", min_pieces_count, 1/100)
              
        save_status_file(result)          
                               
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
            
        agent.exreward = False
        agent.setCache(cache)
        
        agent.show_mcts_process= True
        
        max_emptyCount = random.randint(10,30)
        start_time = time.time()
        mark_reward_piececount = -1
        total_qval=0
        max_qval=0
        min_qval=0
        qval_list=[]
        total_state_value=0
        need_max_ps = False # random.random()>0.5
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
            qval_list.append(qval)
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
                if reward > 0:
                    repeat_count = 40
                    print("#"*repeat_count, 'score:', agent.score, "reward:",reward, 'qval', round(qval,2), 'height:', agent.pieceheight, 'piece:', agent.piececount, \
                        'step:', agent.steps, "step time:", round((time.time()-start_time)/i,3),'reward_p:', agent.piececount-mark_reward_piececount)
                agent.print()
                if agent.piececount%2==0 and (player.need_max_ps or player.need_max_ns):
                    player.need_max_ps = not player.need_max_ps
                    player.need_max_ns = not player.need_max_ns
                
                if agent.pieceheight>8: agent.exreward = False
                                    
                if os.path.exists(self.stop_mark_file):
                    time.sleep(60)
                    raise Exception("find stop mark file")

            # 如果游戏结束或玩了超过1小时或10个方块都没有消除一行
            paytime = time.time()-start_time
                # (agent.removedlines > result["total"]["avg_score"]+1)  or \
            if agent.terminal:# or (agent.state==1 and paytime>60*60):# or agent.piececount-agent.last_reward>=8:
                data["score"] = agent.score
                data["piece_count"] = agent.piececount
                data["piece_height"] = agent.pieceheight
                std_qval = np.std(qval_list)
                if std_qval>100:
                    print("std_qval:", std_qval, qval_list)
                return agent, data, total_qval, total_state_value, max_qval, min_qval, std_qval, start_time, paytime
            


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
          
        print('start test time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        min_removedlines, his_pieces, his_pieces_len = self.test_play(policy_value_net)
        
        print('end test time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # 正式运行
        limit_depth=20
        result = read_status_file() 
        
        if result["total"]["depth"]>limit_depth:
            limit_depth=result["total"]["depth"]
        
        self.n_playout = int(result["total"]["n_playout"])

        self.q_std = result["total"]["q_std"]
        self.q_avg = result["total"]["q_avg"]
        
        if self.q_std>2: self.q_std=2   
        if self.q_std<0.5: self.q_std=0.5
        if self.q_avg>1: self.q_avg=1
        if self.q_avg<-1: self.q_avg=-1       
        
        print("q_std:", self.q_std)   
        player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout, limit_depth=limit_depth)

        cache={}

        # 检查有没有需要重复运行的
        listFiles = [f for f in os.listdir(self.waitplaydir) if f.endswith(".pkl")]
        if listFiles:
            # earliest_file = min(listFiles, key=lambda f: os.path.getctime(os.path.join(self.waitplaydir, f)))
            earliest_file = min(listFiles, key=lambda x: len(x))
            
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
        result = read_status_file() 
        for playcount in range(2):
            player.set_player_id(playcount)
            if playcount==0:
                player.need_max_ps = False
                player.need_max_ns = True
            elif playcount==1:
                player.need_max_ps = True
                player.need_max_ns = False
                
            agent, data, qval, state_value, max_qval, min_qval, std_qval, start_time, paytime = self.play(cache, result, min_removedlines, his_pieces, his_pieces_len, player)
            play_data.append({"agent":agent, "data":data, "qval":qval, "max_qval":max_qval, "min_qval":min_qval, "std_qval":std_qval, "state_value":state_value, "start_time":start_time, "paytime":paytime})
            his_pieces = agent.piecehis
            his_pieces_len = len(agent.piecehis)
                
        print("TRAIN Self Play ending ...")
                
        total_game_score =  play_data[0]["agent"].removedlines + play_data[1]["agent"].removedlines 
        total_game_steps =  play_data[0]["agent"].steps + play_data[1]["agent"].steps 
        total_game_piececount =  play_data[0]["agent"].piececount + play_data[1]["agent"].piececount 
        total_game_paytime =  play_data[0]["paytime"] + play_data[1]["paytime"] 
        total_game_state_value =  play_data[0]["state_value"] + play_data[1]["state_value"] 
        total_game_qval =  play_data[0]["qval"] + play_data[1]["qval"] 
        max_game_qval = max(play_data[0]["max_qval"], play_data[1]["max_qval"])
        min_game_qval = min(play_data[0]["min_qval"], play_data[1]["min_qval"])
        std_game_qval = (play_data[0]["std_qval"] + play_data[1]["std_qval"])/2

        game_score = max(play_data[0]["agent"].removedlines, play_data[1]["agent"].removedlines)
        win_values =[-1, -1]
        if play_data[0]["agent"].piececount>play_data[1]["agent"].piececount:
            win_values[0] = 1
        elif play_data[0]["agent"].piececount<play_data[1]["agent"].piececount:
            win_values[1] = 1
        
        steptime = total_game_paytime/total_game_steps            
        avg_qval = total_game_qval/total_game_steps
        avg_state_value = total_game_state_value/total_game_steps
        
        print("step pay time:", steptime, "qval:", avg_qval, "avg_state_value:", avg_state_value)
        
        result = read_status_file()                       
        alpha = 0.01
        set_status_total_value(result, "score_mcts", total_game_score/2, alpha)
        set_status_total_value(result, "piececount_mcts", total_game_piececount/2, alpha)
        set_status_total_value(result, "q_avg", avg_qval, alpha)
        set_status_total_value(result, "max_qval", max_game_qval, alpha)
        set_status_total_value(result, "min_qval", min_game_qval, alpha)
        set_status_total_value(result, "state_value", avg_state_value, alpha)
        set_status_total_value(result, "step_time", steptime, alpha)
        set_status_total_value(result, "q_std", std_game_qval, alpha)
        
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
            if isinstance(result["best"]["reward"], int):
                result["best"]["reward"] += float(f'0.{result["best"]["reward"]}')
            if result["best"]["reward"]>1:
                result["best"]["reward"] = round(result["best"]["reward"]-1,10) 
    
        save_status_file(result)     
            
        # 计算 acc 看有没有收敛
        pacc = []
        depth = []
        ns = []
        for m, data in enumerate([play_data[0]["data"], play_data[1]["data"]]) :
            for step in data["steps"]:
                pacc.append(step["acc_ps"])
                depth.append(step["depth"])
                ns.append(step["ns"])

        pacc = float(np.average(pacc))
        vdiff = abs(play_data[0]["agent"].piececount-play_data[1]["agent"].piececount)

        depth = float(np.average(depth))
        ns = float(np.average(ns))
        
        result = read_status_file()                       
        set_status_total_value(result, "pacc", pacc, alpha)
        set_status_total_value(result, "vdiff", vdiff, alpha)
        set_status_total_value(result, "depth", depth, alpha)
        set_status_total_value(result, "ns", ns, alpha)  

        update_agent_count = 20
        if result["total"]["_agent"]>update_agent_count:
            result["reward"].append(round(result["total"]["avg_score"],2))
            result["depth"].append(round(result["total"]["depth"],1))
            result["pacc"].append(round(result["total"]["pacc"],2))
            result["vdiff"].append(round(result["total"]["vdiff"],2))
            result["time"].append(round(result["total"]["step_time"],1))
            result["q_avg"].append(round(result["total"]["q_avg"],4))
            result["piececount"].append(round(result["total"]["avg_piececount"],1))
            result["piececount_mcts"].append(round(result["total"]["piececount_mcts"],1))
            result["q_std"].append(round(result["total"]["q_std"],2))
            
            local_time = time.localtime(start_time)
            current_month = local_time.tm_mon
            current_day = local_time.tm_mday

            result["update"].append(current_day)
            result["total"]["_agent"] -= update_agent_count           
            
            # 如果每步的消耗时间小于self.limit_steptime秒，增加探测深度    
            result["total"]["n_playout"] += round(self.limit_steptime-result["total"]["step_time"])
            if result["total"]["n_playout"] < 64: result["total"]["n_playout"] = 64     
            # 保存下中间步骤的agent1
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
        save_status_file(result)         
        avg_qval = [play_data[0]["qval"]/play_data[0]["agent"].steps, play_data[1]["qval"]/play_data[1]["agent"].steps]
        std_qval = [play_data[0]["std_qval"], play_data[1]["std_qval"]]    
        states, mcts_probs, values= [], [], []

        for i, data in enumerate([play_data[0]["data"], play_data[1]["data"]]):     
            if win_values[i]<0:
                v = (win_values[i] * vdiff)/play_data[i]["agent"].piececount
                if v<-1: v=-1    
            else:
                v = 0
                
            for step in data["steps"]:
                states.append(step["state"])
                mcts_probs.append(step["move_probs"])
                if win_values[i]<0:
                    values.append((step["qval"]+v-avg_qval[i])/std_qval[i])
                else:
                    values.append((step["qval"]-avg_qval[i])/std_qval[i])
                
        print("step_values:", values)

        assert len(states)>0
        assert len(states)==len(values)
        assert len(states)==len(mcts_probs)
        
        print("TRAIN Self Play end. length: %s value sum: %s saving ..." % (len(states),sum(values)))

        # 保存对抗数据到data_buffer
        filetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for i, obj in enumerate(self.get_equi_data(states, mcts_probs, values)):
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
        print("max_game_qval:", max_game_qval, "min_game_qval:", min_game_qval, "q_avg:", avg_qval, "std_qval:", std_qval)

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
