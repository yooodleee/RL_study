class EvaluateConfig:
    def __init__(self):
        self.game_num = 400
        self.replace_rate = 0.55
        self.play_config = PlayConfig()
        self.play_config.simulation_num_per_move = 200
        self.play_config.thinking_loop = 1
        self.play_config.c_puct = 1
        self.play_config.change_tau_turn = 0
        self.play_config.noise_eps = 0
        self.evaluate_latest_first = True


class PlayDataConfig:
    def __init__(self):
        self.sl_nb_game_in_file = 100
        self.nb_game_in_file = 100
        self.max_file_num = 200



class PlayConfig:
    def __init__(self):
        self.simulation_num_per_move = 200
        self.thinking_loop = 1
        self.logging_thinking = False
        self.c_puct = 1.5
        self.noise_eps = 0.25
        self.dirichlet_alpha = 0.3
        self.change_tau_turn = 10
        self.virtual_loss = 3
        self.prediction_queue_size = 16
        self.search_threads = 16
        self.prediction_worker_sleep_sec = 0.00001
        self.wait_for_expanding_sleep_sec = 0.000001
        self.resign_threshold = -0.8
        self.min_resign_turn = 5
        self.average_chess_movements = 50



class TrainerConfig:
    def __init__(self):
        self.batch_size = 2048
        self.epoch_to_checkpoint = 1
        self.start_total_steps = 0
        self.save_model_steps = 2000
        self.load_data_steps = 1000
        self.loss_weights = [1.0, 1.0]  # prevent value overfit in SL



class ModelConfig:
    cnn_filter_num = 256
    cnn_filter_size = 3
    res_layer_num = 7
    l2_reg = 1e-4
    value_fc_size = 256
    distributed = True