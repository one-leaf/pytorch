#!/usr/bin/env python3
"""
显示训练状态和历史趋势。
用法: python show_status.py [--json] [--history N]
"""
import json, os, sys

model_name = "vit-ti"
curr_dir = os.path.dirname(os.path.abspath(__file__))
status_file = os.path.join(curr_dir, 'model', model_name, 'status.json')


def fmt(val, decimals=3):
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def show_status(max_history=0, as_json=False):
    if not os.path.exists(status_file):
        print(f"状态文件不存在: {status_file}")
        sys.exit(1)

    with open(status_file, "r") as f:
        state = json.load(f)

    if as_json:
        print(json.dumps(state, indent=2, ensure_ascii=False))
        return

    c = state.get("counters", {})
    m = state.get("metrics", {})
    tr = state.get("training", {})
    info = state.get("info", {})
    history = state.get("history", [])

    # 基本信息
    print("=" * 140)
    print(f"  创建时间:   {info.get('create', '-')}  最后更新:   {info.get('modify', '-')}")
    agent = c.get("agent", 0)
    if agent == 0 and "_agent" in c:
        agent = c["_agent"]
    print(f"  训练轮次:   {agent}")
    print("-" * 140)

    # test_play（纯贪婪，无噪声）
    test_pc = m.get("test_piececount")
    if test_pc is not None and test_pc != 0:
        print("  [Test] test_play（纯贪婪，无噪声）")
        print(f"    平均方块数:   {fmt(test_pc, 1)}    平均步数:     {fmt(m.get('test_steps'), 1)}    平均消行数:   {fmt(m.get('test_removedlines'), 3)}")
        print(f"    历史最高:     方块={m.get('test_piececount_best', 0)}  消行={m.get('test_removedlines_best', 0)}")
    else:
        print("  [Test] （尚未运行 test_play）")
    print("-" * 140)
    print(f"  KL 散度:      {fmt(tr.get('kl'), 6)}")
    print(f"  学习率倍率:   {fmt(tr.get('lr_multiplier'), 4)}")
    train_acc = m.get("train_acc")
    if train_acc is not None and train_acc != 0:
        print(f"  Train EMA:    acc={fmt(train_acc, 4)}  kl={fmt(m.get('train_kl'), 5)}  entropy={fmt(m.get('train_entropy'), 4)}  vloss={fmt(m.get('train_vloss'), 4)}")

    # 历史趋势
    if history and max_history > 0:
        # 保留一头一尾，中间均匀抽样
        if len(history) > max_history:
            if max_history > 2:
                step = (len(history) - 1) / (max_history - 1)
                middle = [history[int(i * step)] for i in range(1, max_history - 1)]
                display = [history[0]] + middle + [history[-1]]
            else:
                display = [history[0], history[-1]]
            label = f"  训练记录 (一头一尾+中间均匀抽样 {max_history}/{len(history)} 条):"
        else:
            display = history
            label = f"  训练记录 ({len(history)} 条):"

        print("=" * 140)
        print(label)
        print("-" * 140)
        header = (f"  {'Agent':>6}  "
                  f"{'GR_Piece':>8} {'GR_Lines':>8} {'GR_Steps':>8} {'GR_Min':>7} {'GR_Max':>7} {'GR_RwdStd':>9}  "
                  f"{'Te_Piece':>8} {'Te_Lines':>8} {'Te_Steps':>8} {'Te_Best':>7}  "
                  f"{'Tr_Acc':>8} {'Tr_KL':>9} {'Tr_Ent':>8} {'Tr_VL':>8}")
        print(header)
        print("-" * 140)
        for h in display:
            print(f"  {h.get('agent', 0):>6}  "
                  f"{h.get('grpo_piececount', 0):>8.1f} "
                  f"{h.get('grpo_removedlines', 0):>8.3f} "
                  f"{h.get('grpo_steps', 0):>8.1f} "
                  f"{h.get('grpo_piececount_min', 0):>7.1f} "
                  f"{h.get('grpo_piececount_max', 0):>7.1f} "
                  f"{h.get('grpo_reward_std', 0):>9.3f}  "
                  f"{h.get('test_piececount', 0):>8.1f} "
                  f"{h.get('test_removedlines', 0):>8.3f} "
                  f"{h.get('test_steps', 0):>8.1f} "
                  f"{h.get('test_piececount_best', 0):>7}  "
                  f"{h.get('train_acc', 0):>8.4f} "
                  f"{h.get('train_kl', 0):>9.5f} "
                  f"{h.get('train_entropy', 0):>8.4f} "
                  f"{h.get('train_vloss', 0):>8.4f}")
        print("=" * 140)
    elif history:
        print("=" * 140)
        print(f"  历史记录: 共 {len(history)} 条 (用 --history N 查看)")
        if len(history) >= 2:
            first, last = history[0], history[-1]
            print(f"  起始(agent {first['agent']}): "
                  f"player pc={first.get('grpo_piececount', 0):.1f} ln={first.get('grpo_removedlines', 0):.3f}  "
                  f"test pc={first.get('test_piececount', 0):.1f} ln={first.get('test_removedlines', 0):.3f}")
            print(f"  当前(agent {last['agent']}):  "
                  f"player pc={last.get('grpo_piececount', 0):.1f} ln={last.get('grpo_removedlines', 0):.3f}  "
                  f"test pc={last.get('test_piececount', 0):.1f} ln={last.get('test_removedlines', 0):.3f}")
        print("=" * 140)


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
