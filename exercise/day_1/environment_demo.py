import random
from environment import Environment


class Agent():
    def __init__(self, env):
        # 環境で用意されているとれるアクションの一覧をエージェントのアクションとする
        self.actions = env.actions

    def policy(self, state):
        # 状況に応じてアクションを選ぶ。今回はランダム
        return random.choice(self.actions)


def main():
    # 環境を作成する
    #  0: ordinary cell
    #  -1: damage cell (game end)
    #  1: reward cell (game end)
    #  9: block cell (can't locate agent)
    grid = [
        [0, 0, 0, 1],
        [0, 9, 0, -1],
        [0, 0, 0, 0]
    ]
    env = Environment(grid)
    agent = Agent(env)

    # 終了するまでアクションをとり続ける試行を10回行う
    for i in range(10):
        state = env.reset()
        total_reward = 0
        is_done = False

        while not is_done:
            # ある状態におけるエージェントのポリシーに従って取るべきアクションを選択する
            action = agent.policy(state)
            # 環境でアクションをとった結果を返す
            next_state, reward, is_done = env.step(action)
            total_reward += reward
            state = next_state

        print('Episode {i}: Agent gets {r} reward.'.format(i=i, r=total_reward))


if __name__ == '__main__':
    main()
