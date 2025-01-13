import re
import shutil

import torch
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

import utils.utils as utils
from models.definitions.DQN import DQN
from utils.constants import *
from utils.replay_buffer import ReplayBuffer
from utils.video_utils import create_gif


if __name__ == "__main__":
    # Step 0: Modify these as needed
    buffer_size = 100000
    epsilon_eval = 0.05
    env_id = "BreakoutNoFrameskip-v4"
    model_name = 'dqn_BreakoutNoFrameskip-v4_ckpt_steps_6810000.pth'
    should_record_video = True

    game_frames_dump_dir = os.path.join(DATA_DIR_PATH, "dqn_eval_dump_dir")
    if os.path.exists(game_frames_dump_dir):
        shutil.rmtree(game_frames_dump_dir)
    os.makedirs(game_frames_dump_dir, exist_ok=True)

    # Step 1: Prepare environment, replay buffer and schedule
    env = utils.get_env_wrapper(env_id, record_video=should_record_video)
    replay_buffer = ReplayBuffer(buffer_size)
    # lambda would als do - doing it like this for consistency
    const_schedule = utils.ConstSchedule(epsilon_eval)    

    # Step 2: Prepare the DQN model
    model_path = os.path.join(BINARIES_PATH, model_name)
    model_state = torch.load(model_path)
    assert model_state["env_id"] == env_id, \
        f"Model {model_name} was trained on {model_state["env_id"]} but you're 
            running it on {env_id}."
    utils.print_model_metadata(model_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn = DQN(env, number_of_actions=env.action_space.n, 
                epsilon_shcedule=const_schedule).to(device)
    dqn.load_state_dict(model_state["state_dict"], strict=True)
    dqn.eval()

    # Step 3: Evaluate the agent on a single episode
    print(f'{"*"*10} Starting the game/ {"*"*10}')
    last_frame = env.reset()

    score = 0
    cnt = 0
    while True:
        replay_buffer.sotre_frame(last_frame)
        # fetch the state, shape = (4, 84, 84) for Atari
        current_state = replay_buffer.fatch_last_state()  

        with torch.no_grad():
            action = dqn.epsilon_greedy(current_state)    # act in this state
        
        # send the action to the environment
        new_frame, reward, done, _ = env.step(action) 
        score += reward

        env.render()    # plot the current game frame
        screen = env.render(mode="rgb_array") # but also save it as an image
        processed_screen = cv.resize(screen[:, :, ::-1], (0, 0), fx=1.5, 
                                     fy=1.5, interpolation=cv.INTER_NEAREST)
        # cv expects BGR hence ::-1
        cv.imwrite(os.path.join(game_frames_dump_dir, 
                                f"{str(cnt).zfill(5)}.jpg"), processed_screen)    
        cnt += 1

        if done:
            print(f"Episode over, score = {score}.")
            break
        
        # see the last frame to the newly acquired frame from the env
        last_frame=new_frame    
    
    create_gif(game_frames_dump_dir, 
               os.path.join(DATA_DIR_PATH, f"{env_id}.gif"), fps=60)