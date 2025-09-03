import math
import numpy as np

from .utils import get_obs, get_mask


N_ACTIONS = 25

class Node:
    def __init__(self, state, to_play, prior=0, parent=None, move=None):
        self.state = state
        self.prior = prior
        self.move = move
        self.to_play = to_play
        self.parent = parent

        self.children = []
        self.is_expanded = False
        self.value_sum = 0.0
        self.visits = 0

    @property
    def is_terminated(self):
        return self.state.is_terminal()

    def value(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

            
    def expand(self, action_probs):
        self.is_expanded = True
        legal = self.state.legal_actions()          
        if not legal:
            return

        priors = np.array([max(0.0, float(action_probs[a])) for a in legal], dtype=np.float32)
        s = priors.sum()
        if s <= 0:
            priors[:] = 1.0 / len(legal) 
        else:
            priors /= s

        for a, p in zip(legal, priors):
            child_state = self.state.clone()
            child_state.apply_action(a)
            self.children.append(Node(
                state=child_state,
                to_play=self.to_play * -1,
                prior=float(p),
                parent=self,
                move=a
            ))


    def select_action(self, temp):
        visit_counts = np.array([child.visits for child in self.children], dtype=np.float32)
        actions = [child.move for child in self.children]

        if temp == 0:
            return actions[int(np.argmax(visit_counts))]

        if temp == float("inf"):
            return np.random.choice(actions)

        visit_counts_dist = visit_counts ** (1.0 / float(temp))
        visit_counts_dist = visit_counts_dist / np.sum(visit_counts_dist)
        return np.random.choice(actions, p=visit_counts_dist)

    def get_action_probs(self, temp=1.0):
        if not self.children:
            return np.zeros(N_ACTIONS, dtype=np.float32) 

        visit_counts = np.array([child.visits for child in self.children], dtype=np.float32)
        actions = [child.move for child in self.children]

        if temp == 0:
            probs = np.zeros(N_ACTIONS, dtype=np.float32)
            probs[actions[int(np.argmax(visit_counts))]] = 1.0
            return probs

        visit_counts_temp = visit_counts ** (1.0 / float(temp))
        visit_counts_temp = visit_counts_temp / np.sum(visit_counts_temp)

        probs = np.zeros(N_ACTIONS, dtype=np.float32)
        for i, a in enumerate(actions):
            probs[a] = visit_counts_temp[i]
        return probs
    
    def select_child(self, c_puct=1.5, eps=1e-8):
        def ucb(child):
            q = 0.0 if child.visits == 0 else -child.value()
            u = c_puct * child.prior * math.sqrt(self.visits + 1.0) / (1.0 + child.visits)
            return q + u
        return max(self.children, key=ucb)



class MCTS:
    def __init__(self, state, model):
        self.state = state
        self.root = Node(state=self.state.clone(),
                         to_play=self.get_player(),
                         prior=0.0)
        self.model = model

    def get_player(self):
        return 1 if self.state.current_player() == 0 else -1

    def _terminal_value_leaf_perspective(self, node: Node) -> float:
        if node.state.player_reward(0) == 1.0:
            winner = 1
        elif node.state.player_reward(1) == 1.0:
            winner = -1
        else:
            return 0.0
        return 1.0 if node.to_play == winner else -1.0

    def run(self, num_simulation: int):
        for _ in range(num_simulation):
            node = self.root

            while node.is_expanded and not node.is_terminated:
                node = node.select_child()

            if not node.is_terminated:
                obs, mask = get_obs(node.state), get_mask(node.state)
                action_probs, value = self.model.play(obs, mask)   
                action_probs, value = action_probs.cpu().numpy(), float(value.item())

                node.expand(action_probs.flatten())

            else:
                value = self._terminal_value_leaf_perspective(node)

            self.backprop(node, value)

        return self.root

    def backprop(self, node: Node, value: float):
        while node is not None:
            node.visits += 1
            node.value_sum += value           
            value = -value                    
            node = node.parent
