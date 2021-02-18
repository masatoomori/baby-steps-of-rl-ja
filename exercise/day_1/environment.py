# @propertyの使い方はこちら
# https://qiita.com/Sylba2050/items/d6f23ac13a0cc5da0c17

# __repr__の使い方はこちら
# __repr__は、可能であれば、Pythonが復元できる(evalで評価すると元のオブジェクトに戻る)、オブジェクトの表現を返すべきと定義されています。
# https://techacademy.jp/magazine/46504

# __eq__の使い方はこちら
# https://www.yoheim.net/blog.php?q=20171002

from enum import Enum
import numpy as np

DAMAGE_CELL = -1
REWARD_CELL = 1
BLOCK_CELL = 9


class State:
    def __init__(self, row=-1, column=-1):
        self.row = row
        self.column = column

    def __repr__(self):
        return '<State: [{r}, {c}]>'.format(r=self.row, c=self.column)

    def clone(self):
        return State(self.row, self.column)

    def __hash__(self):
        return hash((self.row, self.column))

    # クラスのインスタンスが等しいかどうかの定義
    def __eq__(self, other):
        return self.row == other.row and self.column == other.column


class Action(Enum):
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2


class Environment:
    def __init__(self, grid, move_prob=0.8):
        # 二次元のGRIDを引数に取る
        #  0: ordinary cell
        #  -1: damage cell (game end)         -> DAMAGE_CELLという変数にする
        #  1: reward cell (game end)          -> REWARD_CELLという変数にする
        #  9: block cell (can't locate agent) -> BLOCK_CELLという変数にする
        self.grid = grid
        self.agent_state = State()

        # 初期値のRewardは負の値。ゴールに早く行かないといけないことを意味する（？）
        self.default_reward = -0.04

        # Agentは一定の確率で選択した方向に動くが、違う方向に動くこともある
        self.move_prob = move_prob

        self.reset()

    @property
    def row_length(self):
        return len(self.grid)

    @property
    def column_length(self):
        return len(self.grid[0])

    @property
    def actions(self):
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

    @property
    def states(self):
        # 環境の中にあるGRIDの各座標の状態をリストとして返す
        states = []
        for row in range(self.row_length):
            for column in range(self.column_length):
                if self.grid[row][column] != BLOCK_CELL:
                    states.append(State(row, column))
        return states

    # ある状態でアクションをとった際に、次の状態ごとに遷移する確率を返す
    # 指定したアクションであれば所定の確率を割り当てる
    # 上記以外で、指定したアクションと逆方向でなければ、残りの確率を（2方向で）等分する。反対方向には行かない仕様
    # 複数のアクションによって同一の状態に至る可能性がある場合、確率を加算する
    def transit_func(self, state, action):
        transition_prob = {}

        # すでに終端セルにあり、取れるアクションがない場合
        if not self.can_action_at(state):
            return transition_prob

        # 反対方向。To be revised: マイナスをかけるのではなく、Action Classで逆方向を定義すべき
        opposite_direction = Action(action.value * -1)

        for a in self.actions:
            prob = 0
            # 指定したアクションであれば所定の確率を割り当てる
            if a == action:
                prob = self.move_prob
            # 上記以外で、指定したアクションと逆方向でなければ、残りの確率を（2方向で）等分する。反対方向には行かない仕様
            elif a != opposite_direction:
                prob = (1 - self.move_prob) / 2

            # 現在の状態で、各アクションをとった場合に遷移する状態を求める。
            # 複数のアクションによって同一の状態に至る可能性がある場合、確率を加算する
            next_state = self._move(state, a)
            if next_state not in transition_prob:
                transition_prob[next_state] = prob
            else:
                transition_prob[next_state] += prob

        return transition_prob

    def can_action_at(self, state):
        if self.grid[state.row][state.column] == 0:
            return True
        else:
            return False

    def _move(self, state, action):
        # 次の状態は現在の状態をベースとする
        next_state = state.clone()

        if not self.can_action_at(state):
            raise Exception('Cannot move from here!')

        # アクションを実施する
        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.column -= 1
        elif action == Action.RIGHT:
            next_state.column += 1
        else:
            raise Exception('Invalid action selected')

        # 次の状態がGRIDからはみ出ていないか確認する。はみ出ていれば元の状態のまま移動しない
        if next_state.row < 0 or self.row_length <= next_state.row:
            next_state = state
        if next_state.column < 0 or self.column_length <= next_state.column:
            next_state = state

        # BLOCK CELLに移動しようとしていないか確認する。その場合は元の状態のまま移動しない
        if self.grid[next_state.row][next_state.column] == BLOCK_CELL:
            next_state = state

        return next_state

    # 現在の状態を確認し、報酬がもらえるか、終了状態に達したかを判断する。
    # REWORD_CELL, DAMAGE_CELL以外であれば何も起こらない
    def reward_func(self, state):
        reward = self.default_reward
        is_done = False

        attribute = self.grid[state.row][state.column]
        if attribute == REWARD_CELL:
            reward = 1
            is_done = True
        elif attribute == DAMAGE_CELL:
            reward = -1
            is_done = True

        return reward, is_done

    def reset(self):
        # エージェントを左下の状態いセットする
        self.agent_state = State(self.row_length - 1, 0)
        return self.agent_state

    def step(self, action):
        next_state, reward, is_done = self.transit(self.agent_state, action)
        if next_state is not None:
            self.agent_state = next_state

        return next_state, reward, is_done

    def transit(self, state, action):
        transit_prob = self.transit_func(state, action)

        # すでに終端セルにあり、取れるアクションがない場合
        if not transit_prob:
            return None, None, True

        # 次の状態を確率分布に従って選択する
        next_states = []
        probs = []
        for s in transit_prob.keys():
            next_states.append(s)
            probs.append(transit_prob[s])
        next_state = np.random.choice(next_states, p=probs)

        reward, is_done = self.reward_func(next_state)
        return next_state, reward, is_done
