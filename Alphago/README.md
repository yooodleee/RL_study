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

