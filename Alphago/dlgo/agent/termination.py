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


class ResignLargeMargin(TerminationStrategy):
    def __init__(
            self,
            own_color,
            cut_off_move,
            margin):
        
        TerminationStrategy.__init__(self)
        self.own_color = own_color
        self.cut_off_move = cut_off_move
        self.margin = margin

        self.moves_played = 0
    
    def should_pass(self, game_state):
        return False
    
    def should_resign(self, game_state):
        self.moves_played += 1
        if self.moves_played:
            game_result = scoring.compute_game_result(self)
            if game_result.winner != self.own_color and game_result.winning_margin >= self.margin:
                return True
        
        return False


