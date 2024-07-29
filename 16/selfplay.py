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
            result={"reward":[], "depth":[], "pacc":[], "vacc":[], "time":[], "piececount":[]}
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
            result["total"]["exrewardRate"]=0.1  
        if "piececount" not in result:
            result["piececount"]=[]
        if "exrewardRate" in result:
            del result["exrewardRate"]
        if "update" not in result:
            result["update"]=[]
        if "qval" not in result:
            result["qval"]=[]    
        if "rate" not in result:
            result["rate"]=[]    
        if "advantage" not in result:
            result["advantage"]=[]    
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

    def test_play(self,game_json,policy_value_net):
        # 先运行测试
        his_pieces = None
        his_pieces_len = 0
        min_score = 100
        min_removedlines = 0
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
                    result["lastupdate"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    break
                
            self.save_status_file(result, game_json)

            agent.print()

            # 判断是否需要重新玩,如果当前小于平均的0.75，放到运行池训练
            if agent.removedlines < min_score:
                min_score = agent.removedlines
                his_pieces = agent.piecehis
                his_pieces_len = len(agent.piecehis)
                min_removedlines = agent.removedlines
                # filename = "T{}-{}-{}.pkl".format(agent.piececount, agent.removedlines ,int(round(time.time() * 1000000)))
                # savefile = os.path.join(self.waitplaydir, filename)
                # with open(savefile, "wb") as fn:
                #     pickle.dump(his_pieces, fn)
                # print("save need replay", filename)
            
        result["total"]["n_playout"] += (min_removedlines-result["total"]["n_playout"])/100
        self.save_status_file(result, game_json)         
        return min_removedlines, his_pieces, his_pieces_len

    def play(self, cache, result, min_removedlines, his_pieces, his_pieces_len, player, exrewardRate):
        data = {"steps":[],"shapes":[],"last_state":0,"score":0,"piece_count":0}
        if his_pieces!=None:
            print("min_removedlines:", min_removedlines, "pieces_count:", len(his_pieces))
            print("his_pieces:", his_pieces)
            agent = Agent(isRandomNextPiece=True, nextPiecesList=his_pieces)
            agent.is_replay = True
            agent.limitstep = True
        else:
            # 新局按Q值走，探索
            agent = Agent(isRandomNextPiece=True, )
            agent.is_replay = False
            agent.limitstep = False

        agent.setCache(cache)
        
        agent.show_mcts_process= True
        # agent.id = 0 if random.random()>0.5 else 1
        agent.exreward = True #random.random()>0.5
        if agent.exreward:
            agent.exrewardRate = exrewardRate
        else:
            agent.exrewardRate = 1
        
        max_emptyCount = random.randint(10,30)
        start_time = time.time()
        mark_reward_piececount = -1
        avg_qval=0
        avg_state_value=0
        need_max_ps = False # random.random()>0.5
        print("exreward:", agent.exreward,"exrewardRate:", agent.exrewardRate ,"max_emptyCount:",max_emptyCount,"isRandomNextPiece:",agent.isRandomNextPiece,"limitstep:",agent.limitstep,"need_max_ps:",need_max_ps)
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
            need_max_ps = random.random() < agent.removedlines/100   
            action, qval, move_probs, state_value, max_qval, acc_ps, depth, ns = \
                player.get_action(agent, temp=1, need_max_ps=need_max_ps, need_max_qs=agent.is_replay) 

            _, reward = agent.step(action)

            avg_qval += qval

            # if qval > 0:
            #     avg_qval += 1
            # else:
            #     avg_qval += -1

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

            # time.sleep(0.1)

            # 这里的奖励是消除的行数
            if agent.state==1:
                if agent.piececount-agent.last_reward==1:
                    repeat_count = 40
                    print("#"*repeat_count, 'score:', agent.score, "reward:",reward, 'qval', round(qval,2), 'height:', agent.pieceheight, 'piece:', agent.piececount, \
                        'step:', agent.steps, "step time:", round((time.time()-start_time)/i,3),'reward_p:', agent.piececount-mark_reward_piececount)
                agent.print()

            # 如果游戏结束或玩了超过1小时或10个方块都没有消除一行
            paytime = time.time()-start_time
            if agent.terminal or (agent.state==1 and paytime>60*60) or \
                agent.removedlines> result["total"]["avg_score"]+1 or agent.piececount-agent.last_reward>=10:
                data["score"] = agent.score
                data["piece_count"] = agent.piececount
                data["piece_height"] = agent.pieceheight
                return agent, data, avg_qval, avg_state_value, start_time, paytime
            


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
  
        print('start game time:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        min_removedlines, his_pieces, his_pieces_len = self.test_play(game_json, policy_value_net)
        

        # 正式运行
        limit_depth=20
        result = self.read_status_file(game_json) 
        
        if result["total"]["depth"]>limit_depth:
            limit_depth=result["total"]["depth"]
        player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout, limit_depth=limit_depth)

        cache={}

        # if random.random()>0.2:
        his_pieces = None
        his_pieces_len = 0
        min_removedlines = result["total"]["avg_score"]
        for playcount in range(10):
            result = self.read_status_file(game_json) 
            exrewardRate = result["total"]["exrewardRate"]
            
            agent, data, avg_qval, avg_state_value, start_time, paytime = self.play(cache, result, min_removedlines,his_pieces,his_pieces_len,player,exrewardRate)
            
            his_pieces =  agent.piecehis
            his_pieces_len = len(his_pieces)
            
            game_score =  agent.removedlines 
            result = self.read_status_file(game_json)
            
            steptime = paytime/agent.steps            
            avg_qval = avg_qval/agent.steps
            avg_state_value = avg_state_value/agent.steps
            
            print("step pay time:", steptime, "qval:", avg_qval, "avg_state_value:", avg_state_value)
            result["total"]["avg_score_ex"] += (game_score-result["total"]["avg_score_ex"])/100
            result["total"]["avg_reward_piececount"] += (game_score/agent.piececount - result["total"]["avg_reward_piececount"])/1000
                            
            alpha = 0.01
            if exrewardRate==result["total"]["exrewardRate"]:
                result["total"]["avg_qval"] += alpha * (avg_qval - result["total"]["avg_qval"])
                
            result["total"]["avg_state_value"] += alpha * (avg_state_value - result["total"]["avg_state_value"])

            # 速度控制在消耗50行
            if agent.piececount>=his_pieces_len:
                result["total"]["win_count"] += 1
            else:
                result["total"]["lost_count"] += 1
            c = result["total"]["win_count"]+result["total"]["lost_count"]                    
            if c>2000:
                result["total"]["win_count"] -= round(result["total"]["win_count"]/(2*c))
                result["total"]["lost_count"] -= round(result["total"]["lost_count"]/(2*c))
            
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
            for step in data["steps"]:
                pacc.append(step["acc_ps"])
                depth.append(step["depth"])
                ns.append(step["ns"])
                vacc.append(1 if step["qval"]*step["state_value"]>0 else 0)

            pacc = float(np.average(pacc))
            vacc = float(np.average(vacc))
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

            max_list_len=50
            if result["total"]["_agent"]>20:
                result["reward"].append(round(result["total"]["avg_score"],2))
                result["depth"].append(round(result["total"]["depth"],1))
                result["pacc"].append(round(result["total"]["pacc"],2))
                result["vacc"].append(round(result["total"]["vacc"],2))
                result["time"].append(round(result["total"]["step_time"],1))
                result["qval"].append(round(result["total"]["avg_qval"],4))
                result["rate"].append(round(result["total"]["exrewardRate"],5))
                result["piececount"].append(round(result["total"]["avg_piececount"],1))
                if len(result["qval"])>1:
                    result["advantage"].append( (round(result["total"]["exrewardRate"],5), round(result["total"]["avg_qval"]-result["qval"][-2],4)) )
                local_time = time.localtime(start_time)
                current_month = local_time.tm_mon
                current_day = local_time.tm_mday

                result["update"].append(current_month+current_day/100.)
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
                while len(result["rate"])>max_list_len:
                    result["rate"].remove(result["rate"][0])
                while len(result["piececount"])>max_list_len:
                    result["piececount"].remove(result["piececount"][0])
                while len(result["update"])>max_list_len:
                    result["update"].remove(result["update"][0])
                while len(result["advantage"])>max_list_len:
                    result["advantage"].remove(result["advantage"][0])
                    
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

                if len(result["qval"])>5:
                    x=[] #rate
                    y=[] #qval
                    for _x,_y in result["advantage"]:
                        x.append(_x)
                        y.append(_y)
                    # for i in range(len(result["rate"])):
                    #     x.insert(0,result["rate"][i*-1-1])
                    #     y.insert(0,result["qval"][i*-1-1])
                    if len(x)>2:
                        x = np.array(x)
                        y = np.array(y)

                        coefficients = np.polyfit(y, x, deg=1)
                        dst = -1 * result["total"]["avg_qval"]
                        if dst>0.01: dst = 0.01
                        if dst<-0.01: dst = -0.01
                        x_when_y_is_zero = np.polyval(coefficients, dst)
                        # 如果当前平均q值小于0
                        if result["total"]["avg_qval"]<0:
                            if x_when_y_is_zero>result["total"]["exrewardRate"]:
                                result["total"]["exrewardRate"] = x_when_y_is_zero
                            else:
                                if y[-1]<0: # 如果趋势还在减少
                                    result["total"]["exrewardRate"] *= 1.01  
                        elif result["total"]["avg_qval"]>0:
                            if x_when_y_is_zero<result["total"]["exrewardRate"]:                            
                                result["total"]["exrewardRate"] = x_when_y_is_zero
                            else:
                                if y[-1]>0: # 如果趋势还在增加
                                    result["total"]["exrewardRate"] *= 0.99 
                    elif len(result["qval"])>1:    
                        result["total"]["exrewardRate"]+=(result["qval"][-2]-result["qval"][-1])*0.1
                    
                    if result["total"]["exrewardRate"]<1e-4: result["total"]["exrewardRate"]=1e-4
                    if result["total"]["exrewardRate"]>0.1: result["total"]["exrewardRate"]=0.1
                        
            result["lastupdate"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.save_status_file(result, game_json) 
            
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
            print("value: ", pieces_value)
            print("probs: ", pieces_probs)

            print("steps:",step_count,"piece_count:",data["piece_count"],"score:",data["score"],"piece_height:",data["piece_height"])
        
            states, mcts_probs, values, scores= [], [], [], []

            for step in data["steps"]:
                states.append(step["state"])
                mcts_probs.append(step["move_probs"])
                values.append(step["qval"])
                scores.append(step["acc_ps"])
                    
            
            assert len(states)>0
            assert len(states)==len(values)
            assert len(states)==len(mcts_probs)
            assert len(states)==len(scores)

            print("TRAIN Self Play end. length: %s value sum: %s saving ..." % (len(states),sum(values)))


            if playcount==0 or agent.removedlines>min_removedlines: 
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
                # 游戏结束
                if agent.removedlines>min_removedlines: break        
            else:
                print("need replay")
                player.mcts._n_playout=512
            # 删除训练集
            # if agent.piececount/result["total"]["piececount"]<0.5:
            #     filename = "R{}-{}-{}.pkl".format(agent.piececount, agent.score, int(round(time.time() * 1000000)))
            #     his_pieces_file = os.path.join(self.waitplaydir, filename)
            #     print("save need replay", his_pieces_file)
            #     with open(his_pieces_file, "wb") as fn:
            #         pickle.dump(agent.piecehis, fn)
                    

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

