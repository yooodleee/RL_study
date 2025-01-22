from dlgo.goboard import Move


def is_ladder_capture(
        game_state,
        candidate,
        recursion_depth=50):
    
    return is_ladder(
        True,
        game_state,
        candidate,
        None,
        recursion_depth
    )


def is_ladder_escape(
        game_state,
        candidate,
        recursion_depth=50):
    
    return is_ladder(
        False,
        game_state,
        candidate,
        None,
        recursion_depth
    )


