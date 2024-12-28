import matplotlib.pyplot as plt
import numpy 
import seaborn
import torch

import models
from self_play import MCTS, Node, SelfPlay



class DiagnoseModel:
    """
    Tools to understand the learned model.

    Args:
        weights: weights for the model to diagnose.

        configs: configuration class instance related to the weights.
    """

    def __init__(self, checkpoint, config):
        self.config = config

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(checkpoint["weights"])
        self.model.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )   # on GPU if available since the DataParallel objects in MuZeroNetwork requires that
        self.model.eval()

    def get_virtual_trajectory_from_obs(
        self, observation, horizon, plot=True, to_play=0
    ):
        """
        MuZero plays a game but uses its model instead of using the environment.
        Still do an MCTS at each step.
        """
        trajectory_info = Trajectoryinfo("Virtual trajectory", self.config)
        root, mcts_info = MCTS(self.config).run(
            self.model, observation, self.config.action_space, to_play, True
        )
        trajectory_info.store_info(root, mcts_info, None, numpy.nan)

        virtual_to_play = to_play
        for i in range(horizon):
            action = SelfPlay.select_action(root, 0)

            # Players play turn by turn
            if virtual_to_play + 1 < len(self.config.players):
                virtual_to_play = self.config.players[virtual_to_play + 1]
            else:
                virtual_to_play = self.config.players[0]
            
            # Generate new root
            value, reward, policy_logits, hidden_state = self.model.recurrent_inference(
                root.hidden_state,
                torch.tensor([[action]]).to(root.hidden_state.device),
            )
            value = models.support_to_scalar(value, self.config.support_size).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()
            root = Node(0)
            root.expand(
                self.config.action_space,
                virtual_to_play,
                reward,
                policy_logits,
                hidden_state,
            )

            root, mcts_info = MCTS(self.config).run(
                self.model, None, self.config.action_space, virtual_to_play, True, root
            )
            trajectory_info.store_info(
                root, mcts_info, action, reward, new_prior_root_value=value
            )
        
        if plot:
            trajectory_info.plot_trajectory()
        
        return trajectory_info
    
    def compare_virtual_with_real_trajectories(
        self, first_obs, game, horizon, plot=True
    ):
        """
        First, MuZero plays a game but uses its model instead of using the environment.
        Then, MuZero plays the optimal trajectory according precedent trajectory but performs it in the
        real environment until arriving at an action impossible in the real environment.
        It does an MCTS too, but doesn't take it into account.
        All information during the two trajectories are recorded and displayed.
        """
        virtual_trajectory_info = self.get_virtual_trajectory_from_obs(
            first_obs, horizon, False
        )
        real_trajectory_info = Trajectoryinfo("Real trajectory", self.config)
        trajectory_divergence_index = None
        real_trajectory_end_reason = "Reached horizon"

        # Illegal moves are masked at the root
        root, mcts_info = MCTS(self.config).run(
            self.model,
            first_obs,
            game.legal_actions(),
            game.to_play(),
            True,
        )
        self.plot_mcts(root, plot)
        real_trajectory_info.store_info(root, mcts_info, None, numpy.nan)
        for i, action in enumerate(virtual_trajectory_info.action_history):
            # Follow virtual trajecotry until it reaches an illegal move in the real env
            if action not in game.legal_actions():
                break   # Comment to keep playing after trajectory divergence
                action = SelfPlay.select_action(root, 0)
                if trajectory_divergence_index is None:
                    trajectory_divergence_index = i
                    real_trajectory_end_reason = f"Virtual trajectory reached an illegal move at timestep {trajectory_divergence_index}."
            
            observation, reward, done = game.step(action)
            root, mcts_info = MCTS(self.config).run(
                self.model,
                observation,
                game.legal_actions(),
                game.to_play(),
                True,
            )
            real_trajectory_info.store_info(root, mcts_info, action, reward)
            if done:
                real_trajectory_end_reason = "Real trajectory reached Done"
                break
        
        if plot:
            virtual_trajectory_info.plot_trajectory()
            real_trajectory_info.plot_trajectory()
            print(real_trajectory_end_reason)
        
        return (
            virtual_trajectory_info,
            real_trajectory_info,
            trajectory_divergence_index,
        )
    def close_all(self):
        plt.close("all")
    
    def plot_mcts(self, root, plot=True):
        """
        Plot the MCTS!!
        """
        try:
            from graphviz import Digraph
        except ModuleNotFoundError:
            print("Please install graphviz to get the MCTS plot.")
            return None
        
        graph = Digraph(comment="MCTS", engine="neato")
        graph.attr("graph", rankdir="LR", splines="true", overlap="false")
        id = 0

        def traverse(node, action, parent_id, best):
            nonlocal id
            node_id = id
            graph.node(
                str(node_id),
                label=f"Action: {action}\nValue: {node.value():.2f}\nVisit count: {node.visit_count}\nPrior: {node.prior:.2f}\nReward: {node.reward:.2f}",
                color="orange" if best else "black",
            )
            id += 1
            if parent_id is not None:
                graph.edge(str(parent_id), str(node_id), constraint="false")
            
            if len(node.children) != 0:
                best_visit_count = max(
                    [child.visit_count for child in node.children.values()]
                )
            else:
                best_visit_count = False
            for action, child in node.children.items():
                if child.visit_count != 0:
                    traverse(
                        child,
                        action,
                        node_id,
                        True
                        if best_visit_count and child.visit_count == best_visit_count
                        else False
                    )
        
        traverse(root, None, None, True)
        graph.node(str(0), color="red")
        # print(graph.source)
        graph.render("mcts", view=plot, cleanup=True, format="pdf")
        return graph



class Trajectoryinfo:
    """
    Store the information about a trajecotry
    (rewards, search information for every step, ...).
    """

    def __init__(self, title, config):
        self.title = title + ": "