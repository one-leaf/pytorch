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

    total = state.get("total", {})
    info = state.get("info", {})
    history = total.get("history", [])

    # 基本信息
    print("=" * 60)
    print(f"  训练状态  ({model_name})")
    print("=" * 60)
    print(f"  创建时间:   {info.get('create', '-')}")
    print(f"  最后更新:   {info.get('modify', '-')}")

    agent = total.get("agent", 0)
    if agent == 0 and "_agent" in total:
        agent = total["_agent"]
    print(f"  训练轮次:   {agent}")
    print("-" * 60)

    score = total.get("grpo_score")
    pieces = total.get("grpo_piececount")
    steps = total.get("grpo_steps")

    if score is not None:
        print(f"  GRPO 平均消除行数: {fmt(total.get('grpo_removedlines'), 3)}")
        print(f"  GRPO 平均方块数:   {fmt(pieces, 1)}")
        print(f"  GRPO 平均步数:     {fmt(steps, 1)}")
        print(f"  GRPO 最少方块数:   {total.get('grpo_min_piececount', 0)}")
        print(f"  GRPO 最多方块数:   {total.get('grpo_max_piececount', 0)}")
        print(f"  GRPO 最少消除行数: {total.get('grpo_min_removedlines', 0)}")
        print(f"  GRPO 最多消除行数: {total.get('grpo_max_removedlines', 0)}")
    else:
        print(f"  GRPO 分数:         (尚未采集数据)")
    print("-" * 60)
    print(f"  历史最高分:        {total.get('max_score_grpo', 0)}")
    print(f"  历史最高方块数:    {total.get('max_piececount_grpo', 0)}")
    print(f"  历史最低分:        {total.get('min_score_grpo', 0)}")
    print(f"  历史最低方块数:    {total.get('min_piececount_grpo', 0)}")
    print("-" * 60)
    print(f"  KL 散度:           {fmt(total.get('kl'), 6)}")
    print(f"  学习率倍率:        {fmt(total.get('lr_multiplier'), 4)}")

    # 历史趋势
    if history and max_history > 0:
        print("=" * 60)
        print(f"  最近 {min(len(history), max_history)} 条训练记录:")
        print("-" * 60)
        print(f"  {'Agent':>6}  {'Score':>7}  {'Pieces':>7}  {'Lines':>6}  {'Steps':>7}  {'Min':>5}  {'Max':>5}  {'KL':>8}")
        print("-" * 60)
        for h in history[-max_history:]:
            print(f"  {h.get('agent', 0):>6}  {h.get('grpo_score', 0):>7.1f}  "
                  f"{h.get('grpo_piececount', 0):>7.1f}  {h.get('grpo_removedlines', 0):>6.3f}  "
                  f"{h.get('grpo_steps', 0):>7.1f}  "
                  f"{h.get('grpo_min_piececount', 0):>5}  {h.get('grpo_max_piececount', 0):>5}  "
                  f"{h.get('kl', 0):>8.6f}")
        print("=" * 60)
    elif history:
        print("=" * 60)
        print(f"  历史记录: 共 {len(history)} 条 (用 --history N 查看)")
        # 首尾对比
        if len(history) >= 2:
            first, last = history[0], history[-1]
            print(f"  起始(agent {first['agent']}): score={first['grpo_score']:.1f}  "
                  f"pieces={first['grpo_piececount']:.1f}")
            print(f"  当前(agent {last['agent']}): score={last['grpo_score']:.1f}  "
                  f"pieces={last['grpo_piececount']:.1f}")
            print("=" * 60)


if __name__ == '__main__':
    max_hist = 0
    as_json = False
    for arg in sys.argv[1:]:
        if arg == '--json':
            as_json = True
        elif arg == '--history' or arg == '-H':
            max_hist = 10
        elif arg.startswith('-H') and len(arg) > 2:
            max_hist = int(arg[2:])
        elif arg.isdigit():
            max_hist = int(arg)

    if max_hist == 0 and '--history' in sys.argv:
        max_hist = 10
    elif max_hist == 0 and '-H' in sys.argv:
        max_hist = 10

    show_status(max_history=max_hist, as_json=as_json)
