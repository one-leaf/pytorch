"""分析训练数据中方块数的分布，按10个分位桶计算统计量"""
import os, glob, pickle
import numpy as np

model_name = "vit-ti"
curr_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(curr_dir, 'data', model_name)


def load_piececounts():
    """从所有 pkl 文件中提取每局的 piececount"""
    files = glob.glob(os.path.join(data_dir, "*.pkl"))
    piececounts = []
    game_steps_map = {}  # {piececount: [步数列表]}

    for fn in files:
        try:
            with open(fn, "rb") as f:
                steps = pickle.load(f)
            if len(steps) == 0:
                continue
            R = steps[0][6]  # piececount
            n_steps = len(steps)
            piececounts.append(R)
            if R not in game_steps_map:
                game_steps_map[R] = []
            game_steps_map[R].append(n_steps)
        except Exception as e:
            print(f"Error loading {fn}: {e}")

    return np.array(piececounts), game_steps_map


def analyze():
    piececounts, game_steps_map = load_piececounts()

    if len(piececounts) == 0:
        print("No data found!")
        return

    print(f"Total games: {len(piececounts)}")
    print(f"Piececount stats: min={piececounts.min()}, max={piececounts.max()}, "
          f"mean={piececounts.mean():.1f}, median={np.median(piececounts):.1f}, std={piececounts.std():.2f}")
    print()

    # 计算10个分位数边界
    quantiles = np.linspace(0, 100, 11)  # 0%, 10%, 20%, ..., 100%
    boundaries = np.percentile(piececounts, quantiles)

    print("=" * 80)
    print("分位数边界:")
    for i, q in enumerate(quantiles):
        if i < len(boundaries):
            print(f"  Q{int(q):>3d} = {boundaries[i]:.1f}")
    print()

    # 按10个分桶统计
    print("=" * 80)
    print(f"{'桶':>4} {'范围':>14} {'数量':>6} {'占比':>6} {'均值':>6} {'步数均值':>8} {'密度(PDF)':>10} {'归一化权重':>10}")
    print("-" * 80)

    densities = []
    bucket_info = []

    for i in range(10):
        low = boundaries[i]
        high = boundaries[i + 1] if i < 9 else piececounts.max() + 1

        if i < 9:
            mask = (piececounts >= low) & (piececounts < high)
        else:
            mask = (piececounts >= low) & (piececounts <= high)

        count = mask.sum()
        ratio = count / len(piececounts)

        if count > 0:
            bucket_pc = piececounts[mask].mean()
            # 对应步数均值
            bucket_steps = []
            for pc in piececounts[mask]:
                if pc in game_steps_map:
                    bucket_steps.extend(game_steps_map[pc])
            avg_steps = np.mean(bucket_steps) if bucket_steps else 0
        else:
            bucket_pc = (low + high) / 2
            avg_steps = 0

        # 密度 = 占比 / 桶宽度
        width = high - low if high > low else 1
        density = ratio / width
        densities.append(density)

        bucket_info.append({
            'low': low, 'high': high, 'count': count,
            'ratio': ratio, 'mean_pc': bucket_pc, 'avg_steps': avg_steps,
            'density': density
        })

    # 归一化密度（最小=0，最大=1）
    densities = np.array(densities)
    if densities.max() > 0:
        norm_weights = densities / densities.max()
    else:
        norm_weights = densities

    for i, info in enumerate(bucket_info):
        print(f"  {i+1:>2}  [{info['low']:>5.1f},{info['high']:>5.1f}) "
              f"{info['count']:>6} {info['ratio']:>6.3f} "
              f"{info['mean_pc']:>6.1f} {info['avg_steps']:>8.1f} "
              f"{info['density']:>10.4f} {norm_weights[i]:>10.4f}")

    print("-" * 80)
    print()
    print("归一化权重（可直接用作步级别 importance）:")
    print(f"  {norm_weights}")
    print()

    # 可视化
    print("分布可视化:")
    max_bar = 40
    for i, info in enumerate(bucket_info):
        bar_len = int(norm_weights[i] * max_bar)
        bar = "█" * bar_len
        print(f"  Q{i*10:>2}-{(i+1)*10:>2}: {bar} ({norm_weights[i]:.3f})")


if __name__ == '__main__':
    analyze()
