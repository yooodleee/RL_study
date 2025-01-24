## **AlphaGo**
---

### **How to play**
---

#### **Go**

Bot vs. Bot

Run `python bot_v_bot.py` to let 2 Bots play against each other.

Human vs.Bot

Run `python mcts_go.py` to play against a bot.

#### **Tic-Tac-Toe**

Human vs. Bot

Run `pyton play_ttt.py` to play against an unbeatable bot.


### **Reinforcement Learning**
---

1. Run `python init_ac_agent.py --board-size 9 -- output-file ./agent/ac_v1.h5`

2. Run `python self_play_ac.py --board-size 9 -- learning-agent ./agent/ac_v1.h5 --num-games 5000 --experience-out ./experience/exp_0001.h5` to let a bot play against itself and store experiences gaathered during self play.

3. Run `python train_ac.py --learning-agent ./agents/ac_v1.h5 --agent-out ./agents/ac_v2.h5 ./agents/ac_v2.h5 ./--lr 0.01 --bs 1024 experiences/exp_0001.h5` to use experience data for agent improvements via Deep Reingforcement Learning.

4. Run `python eval_ac_bot.py --agent1 ./agents/ac_v2.h5 --agent2 ./agents/ac_v1.h5 --agent2 ./agents/ac_v1.h5 --num-games 100` to check whether the new bot is sronger.

If the new agent is stronger start with it at 2.

Otherwise go to 2. agian to generate more training data. Use multiple experience data files in 3.

Rinse and repeat.

### **Resources**
---

* Paper - Mastering the game of Go with deep neural networks and tree search
* Paper - Mastering the Game of Go without Human Knowledge
* Video - Mastering Games without Human Knowledge
* Book - Deep Learning and the Game of Go
