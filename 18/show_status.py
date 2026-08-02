#!/usr/bin/env python3
"""
显示训练状态和历史趋势。
用法: python show_status.py [--json] [--history N]
"""
import json, os, sys

model_name = ""
curr_dir = os.path.dirname(os.path.abspath(__file__))
play_file = os.path.join(curr_dir, 'model', model_name, 'play.json')
train_file = os.path.join(curr_dir, 'model', model_name, 'train.json')


def fmt(val, decimals=3):
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def show_status(max_history=0, as_json=False):
    play = {}
    train = {}

    if not os.path.exists(play_file) and not os.path.exists(train_file):
        print(f"状态文件不存在: {play_file} 和 {train_file}")
        sys.exit(1)

    if os.path.exists(play_file):
        with open(play_file, "r") as f:
            play = json.load(f)

    if os.path.exists(train_file):
        with open(train_file, "r") as f:
            train = json.load(f)

    if as_json:
        print("play.json:")
        print(json.dumps(play, indent=2, ensure_ascii=False))
        print("\ntrain.json:")
        print(json.dumps(train, indent=2, ensure_ascii=False))
        return

    pc = play.get("counters", {})
    pm = play.get("metrics", {})
    tc = train.get("counters", {})
    tm = train.get("metrics", {})
    tr = train.get("training", {})
    history = train.get("history", [])

    # 基本信息
    print("=" * 153)
    p_info = play.get("info", {})
    t_info = train.get("info", {})
    print(f"  play 创建:   {p_info.get('create', '-')}  最后更新:   {p_info.get('modify', '-')}")
    print(f"  train 创建:  {t_info.get('create', '-')}  最后更新:   {t_info.get('modify', '-')}")
    train_count = tc.get("train", 0)
    sample_count = pc.get("agent", 0)
    print(f"  训练轮次:   {train_count}    样本组数:   {sample_count}")
    print("-" * 153)

    # test_play（纯贪婪，无噪声）
    test_pc = pm.get("test_piececount")
    if test_pc is not None and test_pc != 0:
        print("  [Test] test_play（纯贪婪，无噪声）")
        print(f"    平均方块数:   {fmt(test_pc, 1)}    平均步数:     {fmt(pm.get('test_steps'), 1)}    平均消行数:   {fmt(pm.get('test_removedlines'), 3)}")
        print(f"    历史最高:     方块={pm.get('test_piececount_best', 0)}  消行={pm.get('test_removedlines_best', 0)}")
    else:
        print("  [Test] （尚未运行 test_play）")
    print("-" * 153)
    print(f"  KL 散度:      {fmt(tr.get('kl'), 6)}")
    print(f"  学习率倍率:   {fmt(tr.get('lr_multiplier'), 4)}    熵权重:   {fmt(tr.get('entropy_weight'), 4)}    熵EMA:   {fmt(tr.get('entropy_ema'), 4)}")
    train_acc = tm.get("train_acc")
    if train_acc is not None and train_acc != 0:
        print(f"  Train EMA:    acc={fmt(train_acc, 4)}  kl={fmt(tm.get('train_kl'), 5)}  entropy={fmt(tm.get('train_entropy'), 4)}  vloss={fmt(tm.get('train_vloss'), 4)}")

    # 历史趋势
    if history and max_history > 0:
        # 前期疏后期密：最近 1/3 全显示，前 2/3 均匀抽样 2/3*N 条
        n = len(history)
        if n > max_history:
            recent_n = max(max_history // 3, 2)
            sample_n = max_history - recent_n
            older = history[:n - recent_n]
            recent = history[n - recent_n:]
            if len(older) > sample_n and sample_n > 0:
                step = (len(older) - 1) / (sample_n - 1)
                older_sampled = [older[int(i * step)] for i in range(sample_n)]
            else:
                older_sampled = older
            display = older_sampled + recent
            label = f"  训练记录 (前期疏{len(older_sampled)}条 + 近期密{len(recent)}条，共{max_history}/{n}条):"
        else:
            display = history
            label = f"  训练记录 ({len(history)} 条):"

        print("=" * 153)
        print(label)
        print("-" * 153)
        header = (f"  {'Train':>6}  "
                  f"{'PP_Pc':>8} {'PP_Ln':>8} {'PP_St':>8} {'PP_Min':>7} {'PP_Max':>7}  "
                  f"{'Te_Pc':>8} {'Te_Ln':>8} {'Te_St':>8} {'Te_Best':>7}  "
                  f"{'Acc':>8} {'KL':>9} {'Ent':>8} {'EntW':>7} {'VL':>8}  "
                  f"{'G_M':>7} {'G_S':>7}")
        print(header)
        print("-" * 153)
        for h in display:
            print(f"  {h.get('train', 0):>6}  "
                  f"{h.get('ppo_piececount', 0):>8.1f} "
                  f"{h.get('ppo_removedlines', 0):>8.3f} "
                  f"{h.get('ppo_steps', 0):>8.1f} "
                  f"{h.get('ppo_piececount_min', 0):>7.1f} "
                  f"{h.get('ppo_piececount_max', 0):>7.1f}  "
                  f"{h.get('test_piececount', 0):>8.1f} "
                  f"{h.get('test_removedlines', 0):>8.3f} "
                  f"{h.get('test_steps', 0):>8.1f} "
                  f"{h.get('test_piececount_best', 0):>7}  "
                  f"{h.get('train_acc', 0):>8.4f} "
                  f"{h.get('train_kl', 0):>9.5f} "
                  f"{h.get('train_entropy', 0):>8.4f} "
                  f"{h.get('entropy_weight', 0):>7.3f} "
                  f"{h.get('train_vloss', 0):>8.4f}  "
                  f"{h.get('g_mean_raw', 0):>7.2f} "
                  f"{h.get('g_std_raw', 0):>7.2f}")
        print("=" * 153)
    elif history:
        print("=" * 153)
        print(f"  历史记录: 共 {len(history)} 条 (用 --history N 查看)")
        if len(history) >= 2:
            first, last = history[0], history[-1]
            print(f"  起始(train {first.get('train', 0)}): "
                  f"player pc={first.get('ppo_piececount', 0):.1f} ln={first.get('ppo_removedlines', 0):.3f}  "
                  f"test pc={first.get('test_piececount', 0):.1f} ln={first.get('test_removedlines', 0):.3f}")
            print(f"  当前(train {last.get('train', 0)}):  "
                  f"player pc={last.get('ppo_piececount', 0):.1f} ln={last.get('ppo_removedlines', 0):.3f}  "
                  f"test pc={last.get('test_piececount', 0):.1f} ln={last.get('test_removedlines', 0):.3f}")
        print("=" * 153)


if __name__ == '__main__':
    max_hist = 30  # 默认显示 30 条
    as_json = False
    for arg in sys.argv[1:]:
        if arg == '--json':
            as_json = True
        elif arg == '--history' or arg == '-H':
            max_hist = 30
        elif arg.startswith('-H') and len(arg) > 2:
            max_hist = int(arg[2:])
        elif arg.isdigit():
            max_hist = int(arg)

    if max_hist == 0 and '--history' in sys.argv:
        max_hist = 30
    elif max_hist == 0 and '-H' in sys.argv:
        max_hist = 30

    show_status(max_history=max_hist, as_json=as_json)
