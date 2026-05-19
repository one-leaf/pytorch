import numpy as np

# 5-action space: ROTATION=0, LEFT=1, RIGHT=2, NONE=3, DOWN=4
LEFT_IDX, RIGHT_IDX = 1, 2


def get_equi_data(states, ref_probs, advantages, actions, masks):
    """通过对棋盘左右翻转增加数据集。swap LEFT(idx=1) <-> RIGHT(idx=2)"""
    extend_data = []
    for i in range(len(states)):
        state, ref_prob, adv, action, mask = states[i], ref_probs[i], advantages[i], actions[i], masks[i]
        extend_data.append((state, ref_prob, adv, action, mask))
        # 只在策略对某一侧高度确定时做翻转增强
        if ref_prob[0] < 0.2 and np.max(ref_prob) > 0.8:
            equi_state = np.array([np.fliplr(s) for s in state])
            equi_ref_prob = ref_prob[[0, 2, 1, 3, 4]]
            if action in [LEFT_IDX, RIGHT_IDX]:
                equi_action = 3 - action
            else:
                equi_action = action
            extend_data.append((equi_state, equi_ref_prob, adv, equi_action, mask))
    return extend_data
