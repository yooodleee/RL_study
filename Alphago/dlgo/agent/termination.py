from dlgo import goboard
from dlgo.agent.base import Agent
from dlgo import scoring


class TerminationStrategy:
    def __init__(self):
        pass

    def should_pass(self, game_state):
        return False
    
    def should_resign(self, game_state):
        return False


class PassWhenOpponentPasses(TerminationStrategy):
    def should_pass(self, game_state):
        if game_state.last_move is not None:
            return True if game_state.last_move.is_pass else False


