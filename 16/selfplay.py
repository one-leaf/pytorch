import os, glob, pickle

from model import PolicyValueNet, data_dir, data_wait_dir, model_file
from agent_numba import Agent, ACTIONS
from mcts_single_numba import MCTSPlayer
from collections import deque

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
        self.learn_rate = 1e-8
        self.lr_multiplier = 1.0  # 基于KL的自适应学习率
        self.temp = 1  # MCTS的概率参数，越大越不肯定，训练时1，预测时1e-3
        self.n_playout = 128  # 每个动作的模拟战记录个数，影响后续 512/2 = 256；256/16 = 16个方块 的走法
        # self.min_n_playout = 64   # 最小的模拟战记录个数
        # self.max_n_playout = 256  # 最大1的模拟战记录个数
        # 64/128/256/512 都不行
        # step -> score
        # 128  --> 0.7
        # self.n_playout = 128  # 每个动作的模拟战记录个数，影响后续 128/2 = 66；64/16 = 4个方块 的走法
        self.test_count = 10 # 每次测试次数
        self.play_count = 3 # 每次运行次数
        self.buffer_size = 51200  # cache对次数
        self.play_size = 512  # 每次训练的样本数
        self.epochs = 2  # 每次更新策略价值网络的训练步骤数, 推荐是5
        self.kl_targ = 0.02  # 策略价值网络KL值目标
        self.best_win_ratio = 0.0
        
        # aplhazero 的最佳值是 4 
        # aplhatensor 是 5
        # MCTS child权重， 用来调节MCTS搜索深度，越大搜索越深，越相信概率，越小越相信Q 的程度 默认 5
        # 由于value完全用结果胜负来拟合，所以value不稳，只能靠概率p拟合，最后带动value来拟合
        self.c_puct = self.n_playout/128  # MCTS child权重， 用来调节MCTS中 探索/乐观 的程度 默认 5 
        self.sample_count = 1024  # 每次采样的样本数
        self.split_step_count = 1024  # 每次采样的步数，分割成多个小的样本，默认1024
        self.max_step_count = 10000 
        self.limit_steptime = 1  # 限制每一步的平均花费时间，单位秒，默认1秒

        self.min_piececount = 20  # 限制每局的方块数，少于这个数就认为是失败局面，默认20个方块
        # 等待训练的序列
        self.waitplaydir=os.path.join(data_dir,"replay")
        if not os.path.exists(self.waitplaydir): os.makedirs(self.waitplaydir)

        self.stop_mark_file = os.path.join(self.waitplaydir,"../stop")


    def compute_advantage(self, gamma, lmbda, value_delta):
        # 输入的 td_delta 就是倒装的，输出也是倒装的
        advantage_list = []
        advantage = 0.0
        for delta in value_delta:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        return advantage_list

    def get_equi_data(self, states, mcts_probs, values, advs):
        """
        通过翻转增加数据集
        play_data: [(state, mcts_prob, values, advs), ..., ...]
        """
        extend_data = []
        for i in range(len(states)):
            state, mcts_prob, value, adv=states[i], mcts_probs[i], values[i], advs[i]
            extend_data.append((state, mcts_prob, value, adv))
            if mcts_prob[0]<0.2 and np.max(mcts_prob)>0.8: # 如果旋转的概率的不大，就做翻转
                equi_state = np.array([np.fliplr(s) for s in state])
                equi_mcts_prob = mcts_prob[[0,2,1,3,4]]
                extend_data.append((equi_state, equi_mcts_prob, value, adv))
            # if i==0:
            #     print("state:",state)
            #     print("mcts_prob:",mcts_prob)
            #     print("equi_state:",equi_state)
            #     print("equi_mcts_prob:",equi_mcts_prob)
            #     print("value:",value)
        return extend_data

    def test_play(self,policy_value_net,test_count=None):
        # 先运行测试
        min_his_pieces = None
        min_his_pieces_len = 0
        min_pieces_count = -1
        min_removedlines = -1
        max_pieces_count = -1
        max_removedlines = -1

        state = read_status_file()
        limit_score = 500
        self.min_piececount = state["total"]["min_piececount"] # (state["total"]["piececount"]+state["total"]["min_piececount"])/2     
        no_terminal=0   
        if test_count==None:
            test_count = self.test_count
        for _ in range(test_count):
            agent = Agent(isRandomNextPiece=True)
            start_time = time.time()
            agent.show_mcts_process= False
            # agent.id = 0
            
            for i in range(self.max_step_count):
                action = policy_value_net.policy_value_fn_best_act(agent)
                _, score = agent.step(action)
                if score > 0:
                    print(agent.pieceheight, end=' ')
                    # print("#"*40, 'score:', agent.removedlines, 'height:', agent.pieceheight, 'piece:', agent.piececount, "shape:", agent.fallpiece["shape"], \
                    #     'step:', agent.steps, "step time:", round((time.time()-start_time)/i,3))            
                if agent.terminal : 
                    state = read_status_file()
                    set_status_total_value(state, "score", agent.removedlines, 1/1000)
                    set_status_total_value(state, "piececount", agent.piececount, 1/1000)
                    set_status_total_value(state, "steps", agent.steps, 1/1000)
                    state["lastupdate"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  
                    save_status_file(state)
                    break
                
                if agent.removedlines > limit_score :
                    state = read_status_file()
                    set_status_total_value(state, "score", agent.removedlines, 1/1000)
                    set_status_total_value(state, "piececount", agent.piececount, 1/1000)
                    set_status_total_value(state, "steps", agent.steps, 1/1000)
                    state["lastupdate"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  
                    save_status_file(state)
                    no_terminal += 1
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
                
            if agent.piececount<self.min_piececount:
                filename = "{:05d}-{:05d}-{}.pkl".format(min_his_pieces_len, min_removedlines, "".join(min_his_pieces)[:50])
                his_pieces_file = os.path.join(self.waitplaydir, filename)
                print("save need replay", his_pieces_file)
                with open(his_pieces_file, "wb") as fn:
                    pickle.dump(min_his_pieces, fn)
                    
        state = read_status_file()
        
        set_status_total_value(state, "max_score", max_removedlines, 1/100)
        set_status_total_value(state, "max_piececount", max_pieces_count, 1/100)
        set_status_total_value(state, "min_score", min_removedlines, 1/100)
        set_status_total_value(state, "min_piececount", min_pieces_count, 1/100)
        set_status_total_value(state, "no_terminal_rate", no_terminal/test_count, 1/100)
              
        save_status_file(state)          
                               
        return min_removedlines, min_his_pieces, min_his_pieces_len

    def play(self, cache, state, sample_count, his_pieces, his_pieces_len, player, policy_value_net):
        # data = {"steps":deque(maxlen=self.play_size),"last_state":0,"score":0,"piece_count":0}
        data = {"steps":[],"last_state":0,"score":0,"piece_count":0}
        if his_pieces_len>0:
            print("sample_count:", sample_count, "pieces_count:", len(his_pieces))
            print("his_pieces:", his_pieces)
            agent = Agent(isRandomNextPiece=False, nextPiecesList=his_pieces)
        else:
            agent = Agent(isRandomNextPiece=True)

        agent.show_mcts_process= False
        for i in count():
            action = policy_value_net.policy_value_fn_best_act(agent)
            _, score = agent.step(action)
            if agent.terminal:
                break
            if agent.removedlines > 1000:
                raise Exception("removedlines too large, cancel play")
            
        agent.print()    
        print("agent.steps:", agent.steps, "agent.piececount:", agent.piececount, "agent.removedlines:", agent.removedlines)
        his_pieces = agent.piecehis
        his_pieces_len = len(his_pieces)
        his_steps = agent.steps

        # 新局按Q值走，探索
        agent = Agent(isRandomNextPiece=False, nextPiecesList=his_pieces )
        agent.is_replay = False
        agent.limitstep = False

        # if his_steps > sample_count:
        #     while True:
        #         action = policy_value_net.policy_value_fn_best_act(agent)
        #         agent.step(action)        
        #         if agent.steps > (his_steps-sample_count) and agent.state==1:
        #             break
        #         if agent.terminal:
        #             raise Exception("agent terminal, cancel play")
                
        # agent.print()    
                
        # agent.piececount = 0
        # agent.steps = 0
        # agent.removedlines=0  
        agent.setCache(cache)
        
        agent.show_mcts_process= True
        
        max_emptyCount = random.randint(10,30)
        start_time = time.time()
        total_qval=0
        avg_qval=0
        qval_list=[]
        total_state_value=0
        find_end_steps = 0
        has_find_end = False
        player.need_max_ps = not player.need_max_ns
        print("max_emptyCount:",max_emptyCount,"isRandomNextPiece:",agent.isRandomNextPiece,"limitstep:",agent.limitstep,"max_ps:",player.need_max_ps,"max_qs:",agent.is_replay)
        for i in count():
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
            action, qval, move_probs, state_value, acc_ps, depth, find_end = player.get_action(agent, temp=1) 

            _, score = agent.step(action)

            total_qval += qval 
            # if qval > 0:
            #     avg_qval += 1
            # else:
            #     avg_qval += -1

            total_state_value += state_value 

            _step["piece_height"] = agent.pieceheight
            _step["score"] = score if score>0 else 0
            _step["move_probs"] = move_probs
            _step["state_value"] = state_value
            _step["qval"] = qval
            _step["acc_ps"] = acc_ps
            _step["depth"] = depth
            _step["find_end"] = find_end
            
            data["steps"].append(_step)

            if find_end: has_find_end = True
            if has_find_end: find_end_steps += 1
            # time.sleep(0.1)

            # 这里的奖励是消除的行数
            if agent.state==1:
                if score > 0:
                    repeat_count = 40
                    print("#"*repeat_count,  "score:",score, 'qval', round(qval,2), 'height:', agent.pieceheight, 'piece:', agent.piececount, \
                        'step:', agent.steps, "step time:", round((time.time()-start_time)/(i+1),3))
                agent.print()
                # if agent.piececount%2==0 and (player.need_max_ps or player.need_max_ns):
                player.need_max_ps = not player.need_max_ps
                player.need_max_ns = not player.need_max_ns
                                                    
                if os.path.exists(self.stop_mark_file):
                    print("stop mark file found, exit after waiting 60s")
                    time.sleep(60)
                    raise Exception("find stop mark file")

            # 如果游戏结束或玩了超过2小时
            paytime = time.time()-start_time
                # (agent.removedlines > state["total"]["avg_score"]+1)  or \

            if agent.terminal:

                # # 修复Q值，将最后都无法消行的全部设置为-1
                # if agent.terminal:
                #     score_count = 0
                #     for i in range(len(data["steps"])-1,-1,-1):
                #         if data["steps"][i]["score"]>0: score_count += 1
                #         # if score_count>10: break
                #         if score_count == 0:
                #             data["steps"][i]["qval"] -= 1
                #         else:
                #             data["steps"][i]["qval"] -= 1/score_count
                #         # data["steps"][i]["qval"] -= 0.9**(agent.piececount-data["steps"][i]["piece_count"])

                data["score"] = agent.removedlines
                data["piece_count"] = agent.piececount
                data["steps_count"] =  agent.steps
                data["find_end_steps"] = find_end_steps
                
                qval_list = [step["qval"] for step in data["steps"]]
                # avg_first = np.average(qval_list)
                # while len(qval_list)<self.play_size:
                #     qval_list.insert(0,avg_first) 
                std_qval = float(np.std(qval_list))
                avg_qval = float(np.average(qval_list))
                
                # print_v = (np.array(qval_list)-avg_qval)/std_qval     
                
                # print(print_v[:200])
                # print("...")
                # print(print_v[-200:])
                print(np.array(qval_list))
                
                return agent, data, total_qval, total_state_value, avg_qval, std_qval, start_time, paytime


    def collect_selfplay_data(self):
        """收集自我对抗数据用于训练"""       
        print("TRAIN Self Play starting ...")
        
        # 这里用上一次训练的模型作为预测有助于模型稳定
        load_model_file=model_file
        if os.path.exists(model_file+".bak"):
            load_model_file = model_file+".bak"

        if os.path.exists(load_model_file):
            if time.time()-os.path.getmtime(load_model_file)>60*60*5:
                print("超过5小时模型都没有更新了，停止训练")
                time.sleep(60)
                return

        # 开始游戏
        policy_value_net = PolicyValueNet(GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=load_model_file)
        bestmodelfile = model_file+"_best"
          
        print('start test time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        min_removedlines, his_pieces, his_pieces_len = self.test_play(policy_value_net)
        
        print('end test time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # 正式运行
        state = read_status_file() 
        
        
        # self.n_playout = int(state["total"]["n_playout"])
        # self.sample_count = int((state["total"]["steps"]+state["total"]["steps_mcts"])//2)       
        
        self.sample_count = state["total"]["find_end_steps"]*2
        # if self.sample_count < state["total"]["steps_mcts"]:
        #     self.sample_count = state["total"]["steps_mcts"]
            
        # if self.sample_count < 520: self.sample_count = 520
        # if self.sample_count > state["total"]["steps_mcts"]:
        #     self.sample_count = state["total"]["steps_mcts"]
        self.sample_count = int(self.sample_count)          
        print("sample_count:", self.sample_count)   
        
        player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout, limit_count=self.sample_count, min_score=state["total"]["min_score"])

        cache={}

        # 检查有没有需要重复运行的
        listFiles = [f for f in os.listdir(self.waitplaydir) if f.endswith(".pkl")]
        if listFiles :#and random.random()>0.20:
            earliest_files = sorted(listFiles, key=lambda f: os.path.getctime(os.path.join(self.waitplaydir, f)))
            # earliest_files = sorted(listFiles)
            
            # while len(earliest_files)>200:
            #     newmodelfile = earliest_files.pop()
            #     os.remove(os.path.join(self.waitplaydir, newmodelfile))
                
            filename = os.path.join(self.waitplaydir, earliest_files[0])
            try:
                with open(filename, "rb") as fn:
                    his_pieces = pickle.load(fn)
                    his_pieces_len = len(his_pieces)
                print("load need replay", filename)
            finally:
                os.remove(filename)
        else:
            his_pieces = []
            his_pieces_len = 0
        
        play_data = []
        state = read_status_file() 
        need_replay = True
        for playcount in range(self.play_count):
            player_id = random.randint(0, 2)
            player.set_player_id(player_id)
                
            agent, data, qval, state_value, avg_qval, std_qval, start_time, paytime = self.play(cache, state, self.sample_count, his_pieces, his_pieces_len, player, policy_value_net)
                        
            play_data.append({"agent":agent, "data":data, "qval":qval, "avg_qval":avg_qval, "std_qval":std_qval, "state_value":state_value, "start_time":start_time, "paytime":paytime})
            his_pieces = agent.piecehis
            his_pieces_len = len(agent.piecehis)
            
            # 如果游戏达到了最小的消除行数，样本有效，直接结束
            # if agent.steps >= self.sample_count:
            # # if his_pieces_len > agent.next_Pieces_list_len:
            #     self.play_count = playcount+1
            #     need_replay = False
            #     break
                            
        print("TRAIN Self Play ending ...")
        if need_replay and state["total"]["min_score"]>1:
            min_his_pieces = play_data[-1]["agent"].piecehis
            min_his_pieces_len = len(play_data[-1]["agent"].piecehis)                    
            filename = "{:05d}-{:05d}-{}.pkl".format(min_his_pieces_len, min_removedlines, "".join(min_his_pieces)[:50])
            his_pieces_file = os.path.join(self.waitplaydir, filename)
            print("save need replay", his_pieces_file)
            with open(his_pieces_file, "wb") as fn:
                pickle.dump(min_his_pieces, fn)            
                
        total_game_score =  sum([play_data[i]["data"]["score"] for i in range(self.play_count)]) 
        total_game_steps =  sum([play_data[i]["data"]["steps_count"] for i in range(self.play_count)])  
        total_game_piececount =  sum([play_data[i]["data"]["piece_count"] for i in range(self.play_count)]) 
        total_game_paytime =  sum([play_data[i]["paytime"] for i in range(self.play_count)])  
        total_game_state_value =  sum([play_data[i]["state_value"] for i in range(self.play_count)])
        total_game_qval =  sum([play_data[i]["qval"] for i in range(self.play_count)]) 
        max_game_qval = max([play_data[i]["avg_qval"] for i in range(self.play_count)]) 
        min_game_qval = min([play_data[i]["avg_qval"] for i in range(self.play_count)]) 
        std_game_qval = sum([play_data[i]["std_qval"] for i in range(self.play_count)]) / self.play_count
        find_end_steps = sum([play_data[i]["data"]["find_end_steps"] for i in range(self.play_count)]) / self.play_count

        probs_list = []
        for i in range(self.play_count):
            probs_list.extend([play_data[i]["data"]["steps"][j]["move_probs"] for j in range(len(play_data[i]["data"]["steps"]))])
        std_game_prob = np.std(np.array(probs_list), axis=1).mean()

        game_score = max([play_data[i]["data"]["score"] for i in range(self.play_count)])
                
        steptime = total_game_paytime/total_game_steps            
        avg_qval = total_game_qval/total_game_steps
        avg_state_value = total_game_state_value/total_game_steps
        avg_game_score = total_game_score/self.play_count
        avg_game_piececount = total_game_piececount/self.play_count
        
        print("step pay time:", steptime, "qval:", avg_qval, "avg_state_value:", avg_state_value)

        win_values =[-1 for i in range(self.play_count)]
        for i in range(self.play_count):
            if play_data[i]["data"]["piece_count"] > avg_game_piececount:
                win_values[i] = 1
        
        state = read_status_file()                       

        if len(state["total"]["win_lost_tie"])!=self.play_count:
            state["total"]["win_lost_tie"]=[0 for _ in range(self.play_count)]
        for i in range(self.play_count):
            if win_values[i]==1:
                state["total"]["win_lost_tie"][i] += 1

        alpha = 0.01
        set_status_total_value(state, "score_mcts", avg_game_score, alpha)
        set_status_total_value(state, "piececount_mcts", avg_game_piececount, alpha)
        set_status_total_value(state, "q_avg", avg_qval, alpha)
        set_status_total_value(state, "step_time", steptime, alpha)
        set_status_total_value(state, "q_std", std_game_qval, alpha)
        set_status_total_value(state, "p_std", std_game_prob, alpha)
        # increments = [state["steps_mcts"][i] - state["steps_mcts"][i - 1] for i in range(1, len(state["steps_mcts"]))]
        # avg_increments = np.mean(increments) if len(increments)>0 else 0       
        # set_status_total_value(state, "steps_mcts", total_game_steps/self.play_count - avg_increments, alpha)
        set_status_total_value(state, "steps_mcts", total_game_steps/self.play_count, alpha)
        set_status_total_value(state, "find_end_steps", find_end_steps, alpha)
                
        state["total"]["agent"] += 1
        state["total"]["_agent"] += 1
    
        if game_score>state["best"]["score"]:
            state["best"]["score"] = game_score
            state["best"]["agent"] = state["total"]["agent"]
        else:
            if isinstance(state["best"]["score"], int):
                state["best"]["score"] += float(f'0.{state["best"]["score"]}')
            if state["best"]["score"]>1:
                state["best"]["score"] = round(state["best"]["score"]-1,10) 
    
        save_status_file(state)     
            
        # 计算 acc 看有没有收敛
        pacc = []
        depth = []
        for data in [play_data[i]["data"] for i in range(self.play_count)] :
            for step in data["steps"]:
                pacc.append(step["acc_ps"])
                depth.append(step["depth"])

        pacc = float(np.average(pacc))
 
        depth = float(np.average(depth))
        
        state = read_status_file()                       
        set_status_total_value(state, "pacc", pacc, alpha)
        set_status_total_value(state, "depth", depth, alpha)
        
        update_agent_count = 20
        if state["total"]["_agent"]>update_agent_count:
            state["score"].append(round(state["total"]["score"]))
            state["score_mcts"].append(round(state["total"]["score_mcts"]))
            state["depth"].append(round(state["total"]["depth"],1))
            state["pacc"].append(round(state["total"]["pacc"],2))
            state["step_time"].append(round(state["total"]["step_time"],1))
            state["q_avg"].append(round(state["total"]["q_avg"],2))
            state["piececount"].append(round(state["total"]["piececount"]))
            state["piececount_mcts"].append(round(state["total"]["piececount_mcts"]))
            state["q_std"].append(round(state["total"]["q_std"],2))
            state["p_std"].append(round(state["total"]["p_std"],2))
            state["min_score"].append(round(state["total"]["min_score"]))
            state["no_terminal_rate"].append(round(state["total"]["no_terminal_rate"],2))
            state["steps_mcts"].append(round(state["total"]["steps_mcts"]))
            state["steps"].append(round(state["total"]["steps"]))
            state["find_end_steps"].append(round(state["total"]["find_end_steps"]))
            
            local_time = time.localtime(start_time)
            current_month = local_time.tm_mon
            current_day = local_time.tm_mday

            state["update"].append(current_day)
            state["total"]["_agent"] -= update_agent_count           
            # state["total"]["sample_depth"] += (self.sample_count-state["total"]["steps_mcts"])*0.01
            # 如果每步的消耗时间小于self.limit_steptime秒，增加探测深度    
            # if len(state["score"])>=5:
            #     x = np.arange(5)
            #     slope, intercept = np.polyfit(x, state["score"][-5:], 1)
            #     eps = 1e-6
            #     if slope > eps:
            #         print("score 趋势:", slope, "正在上升")
            #         state["total"]["n_playout"] -= 1
            #     elif slope < -eps:
            #         print("score 趋势:", slope, "正在下降")
            #         # 如果奖励在下降，适当增加探索深度
            #         state["total"]["n_playout"] += 1
            #     if state["total"]["n_playout"] < self.min_n_playout: state["total"]["n_playout"] = self.min_n_playout 
            #     if state["total"]["n_playout"] > self.max_n_playout: state["total"]["n_playout"] = self.max_n_playout                         
                        
                # else:
                #     print("score 趋势:", slope, "没有变化")
                #     # 如果没有变化，适当降低探索深度
                #     state["total"]["n_playout"] += round(self.limit_steptime-state["total"]["step_time"])
                #     if state["total"]["n_playout"] < self.min_n_playout: state["total"]["n_playout"] = self.min_n_playout   
                      
            # 保存下中间步骤的agent1
            # newmodelfile = model_file+"_"+str(state["total"]["agent"])
            # if not os.path.exists(newmodelfile):
            #     policy_value_net.save_model(newmodelfile)

            # 如果当前最佳，将模型设置为最佳模型
            if max(state["score"])==state["score"][-1]:
                newmodelfile = model_file+"_score_"+str(state["score"][-1])
                if not os.path.exists(newmodelfile):
                    policy_value_net.save_model(newmodelfile)
                if os.path.exists(bestmodelfile): os.remove(bestmodelfile)
                if os.path.exists(newmodelfile): os.link(newmodelfile, bestmodelfile)
                        
        state["lastupdate"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        save_status_file(state)         
        avg_qval_list = [play_data[i]["avg_qval"] for i in range(self.play_count)]
        std_qval_list = [play_data[i]["std_qval"] for i in range(self.play_count)]    
        states, mcts_probs, values, advs= [], [], [], []

        # 将Q值转为优势A
        # 1 用 Q_i = (Q_i - mean(Q))/std(Q) 转为均衡Q
        # 2 用 A_i = Q_i+1 - Q_i 转为优势 A
        
        split_step_count = self.split_step_count
        print("split_step_count:", split_step_count)
        for i in range(self.play_count):
            len_steps = len(play_data[i]["data"]["steps"])           
            
            qval_list=np.zeros(split_step_count, dtype=np.float32)
            adv_list=np.zeros(split_step_count, dtype=np.float32)
            rem = len_steps%split_step_count   
            t = len_steps//split_step_count
            
            # v = np.linspace(-1, 1, len_steps, dtype=np.float32)
            # v = v/np.std(v)
            # v = np.clip(v, -1, 1)
            # values.extend(v.tolist())

            c = 0
            for k in range(len_steps-1, -1, -1):
                step = play_data[i]["data"]["steps"][k]
                
                c_rem = c%split_step_count
                if c>0 and c_rem==0:
                    # qval_list += qvals[k:k+split_step_count][::-1]
                    qval_mean = np.mean(qval_list)#-1/play_data[i]["data"]["piece_count"]
                    qval_std = np.std(qval_list)+1e-6
                    adv_mean = np.mean(adv_list)
                    adv_std = np.std(adv_list)+1e-6
                    qval_list = (qval_list - qval_mean) / qval_std
                    adv_list = (adv_list - adv_mean) / adv_std
                    qval_list = np.clip(qval_list, -1, 1)
                    adv_list = np.clip(adv_list, -1, 1)
                    values.extend(qval_list.tolist())
                    advs.extend(adv_list.tolist())
                    print(i, "qval_mean:", qval_mean, "adv_mean:", adv_mean, "adv_std:", adv_std)
                    print(qval_list[::-1])
                    print(adv_list[::-1])                        
                    qval_list[:]=0    
                    adv_list[:]=0

                qval_list[c_rem]=step["qval"]
                # 这里用 Q_t+1 - Q_t 转为优势A
                if k==0:
                    adv_list[c_rem] = 0# step["qval"] - 0
                else:
                    adv_list[c_rem] = step["qval"] - play_data[i]["data"]["steps"][k-1]["qval"]
                # adv_list[c_rem]=step["qval"] - step["state_value"]        
                c += 1
                
                states.append(step["state"])
                mcts_probs.append(step["move_probs"])
            
            if t==0 or rem>=split_step_count//2:
                # print(qval_list[:rem])
                # print(adv_list[:rem])
                # qval_list += qvals[:rem][::-1]
                qval_mean = np.mean(qval_list[:rem])#-1/play_data[i]["data"]["piece_count"]
                qval_std = np.std(qval_list[:rem])+1e-6
                adv_mean = np.mean(adv_list[:rem])
                adv_std = np.std(adv_list[:rem])+1e-6
                qval_list = (qval_list - qval_mean) / qval_std
                adv_list = (adv_list - adv_mean) / adv_std
                qval_list = np.clip(qval_list, -1, 1)
                adv_list = np.clip(adv_list, -1, 1)
                values.extend(qval_list[:rem].tolist())
                advs.extend(adv_list[:rem].tolist())
                print(i, "qval_mean:", qval_mean, "qval_std:", qval_std, "adv_mean:", adv_mean, "adv_std:", adv_std)
                print(qval_list[:rem][::-1])
                print(adv_list[:rem][::-1])
                
                    
                
        # 2 用 Q_t+1 - Q_t 转为优势A
        # for i in range(self.play_count):
        #     len_steps = len(play_data[i]["data"]["steps"])
        #     data = [play_data[i]["data"]["steps"][k]["qval"] for k in range(len_steps)]
            
        #     for k in range(len_steps):
        #         step = play_data[i]["data"]["steps"][k]
        #         step["qval"] = (np.mean(data[k:k+4]) - step["qval"])
                # if k==len_steps-1:
                #     step["qval"] = -1
                # else:
                #     step["qval"] = play_data[i]["data"]["steps"][k+1]["qval"] - step["qval"]           

        # _temp_values = deque(maxlen=self.sample_count)
        # for i in range(self.play_count):
        #     # 如果当前局面有足够的步数，跳过
        #     len_steps = len(play_data[i]["data"]["steps"])
        #     if len_steps>=self.sample_count:
        #         # for k in range(len_steps-self.sample_count, len_steps):
        #         for k in range(len_steps):
        #             step = play_data[i]["data"]["steps"][k]
        #             _temp_values.append(step["qval"]) 
        #         mean_val = np.mean(_temp_values)
        #         std_val = np.std(_temp_values)
        #         # mean_val = 0
        #         # std_val = 1
                    
        #         # for k in range(len_steps-self.sample_count, len_steps):
        #         for k in range(len_steps):
        #             step = play_data[i]["data"]["steps"][k]
        #             states.append(step["state"])
        #             mcts_probs.append(step["move_probs"])
        #             values.append((step["qval"]-mean_val)/std_val)
        #     else: 
        #         # 如果当前局面步数不够，全部补充1到 self.sample_count 计算方差
        #         for k in range(self.sample_count-len_steps):
        #             _temp_values.append(1)
        #         for step in play_data[i]["data"]["steps"]:
        #             _temp_values.append(step["qval"])
        #         mean_val = np.mean(_temp_values)
        #         std_val = np.std(_temp_values)                
        #         # mean_val = 0
        #         # std_val = 1
        #         for step in play_data[i]["data"]["steps"]:                    
        #             states.append(step["state"])
        #             mcts_probs.append(step["move_probs"])
        #             values.append((step["qval"]-mean_val)/std_val)
                    
                    # values.append(step["qval"]+play_data[i]["agent"].piececount/total_game_piececount) 
        # values = np.array(values, dtype=np.float16)
        # # 计算均值和标准差
        # mean_val = np.mean(values)
        # std_val = np.std(values)
        # # 标准化
        # normalized = (values - mean_val) / std_val
        # # 裁剪到 [-1, 1]
        # values = np.clip(normalized, -1, 1)    
        # values = values/np.std(values)
        # values = np.clip(values, -1, 1)

        assert len(states)>0
        assert len(states)==len(values)
        assert len(states)==len(mcts_probs)
        assert len(states)==len(advs)
        
        states_len = len(states)
            
        print("avg_values:", np.mean(values), "std_values:", np.std(values))
        
        print("TRAIN Self Play end. length: %s value sum: %s saving ..." % (states_len,sum(values)))

        # 保存对抗数据到data_buffer
        filetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for i, obj in enumerate(self.get_equi_data(states, mcts_probs, values, advs)):
        # for i, obj in enumerate(zip(states, mcts_probs, values, advs)):
            filename = "{}-{}.pkl".format(filetime, i)
            savefile = os.path.join(data_wait_dir, filename)
            with open(savefile, "wb") as fn:
                pickle.dump(obj, fn)
        print("saved file basename:", filetime, "length:", i+1)
        print()
        # for i in range(self.play_count):
        #     print("##################", i, "##################")
        #     play_data[i]["agent"].print()
        #     print()
        print("win_values:", win_values)
        print("score:", [play_data[i]["data"]["score"] for i in range(self.play_count)])
        print("steps:", [play_data[i]["data"]["steps_count"] for i in range(self.play_count)])
        print("piececount:", [play_data[i]["data"]["piece_count"] for i in range(self.play_count)])
        print("paytime:", [play_data[i]["paytime"] for i in range(self.play_count)])
        print("max_game_qval:", max_game_qval, "min_game_qval:", min_game_qval, "q_avg:", avg_qval_list, "std_qval:", std_qval_list)

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
