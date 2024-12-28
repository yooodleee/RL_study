## **About**

Chess reinforcement learning by AlphaGo Zero methods.

## **Environment**

* Python 3.6.3+
* tensorflow-gpu: 1.3.0
* Keras: 2.0.8

## **Modules**

**Supervised Learning**

To use the new SL process is as simple as running in the beggining instead of the worker "self" the new worker "sl".
Onece the model converges enough with SL play-data, just stop the worker "sl" and start the worker "self" so the model will start improving now due to self-play data.

If you want to use this new SL step, you will have to download big PGN files (chess files) and paste them into the `data/play_data` folder (FICS is a good source of data). You can also use the SCID program to filter by headers like player ELO, game result and more.

To avoid overfitting, recommend using data setes of at least 3000 games and running at most 3-4 epochs.

**Reinforcement Learning**

This AlphaGo Zero implementation consists of three workers:
`self`, `opt`, `eval`.

* `self` is Self-Play to generate training data by self-play using BestModel.
* `opt` is Trainer to train model, and generate next-generation models.
* `eval` is Evaluator to evaluate whether the next-generation model is better than BestModel. If better, replace BestModel.

**Distributed Training**

Now it's possible to train the model in a distributed way. The only thing needed is to use the new parameter:

* `--type distributed`: use mini config for testing, 
(see `see/chess_zero/configs/distributed.py`)

So, in order to contribute to the distributed team you just need to run the three workers locally like this:

```
python src/chess_zero/run.py self --type distributed
(or python src/chess_zero/run.py sl --type distributed)
python src/run/chess_zero/run.py opt --type distributed
python src/run/chess_zero/run.py eval --type distributed
```

**GUI**

* `uci` launches the Universal Chess Interface, for use in a GUI.

To set up ChessZero with GUI, point it to `c0uci.bat` (or rename to .sh). For example, this is screenshot of the random model using Arena's self-play feature

**Data**

* `data/model/model_best_*`: BestModel
* `data/model/next_generation/*`: next-generation models.
* `'data/play_data/play_*.json`: generated training data.
* `logs/main.log`: log file.

## **How to use**

### **Setup**

**install libraries**

```
pip install -r requirements.txt
```

If you want to use GPU:
```
pip install tensorflow-gpu
```

Make sure Keras is using TensorFlow and you have Python 3.6.3+

### **Basic Usage**

For training model, execute `self-play`, `Trainer` and `Evaluator`.

### **Self-Play**

```
python src/chess_zero/run.py self
```

When executed, Self-Play will start using BestModel, If the BestModel does not exist, new random model will be created and become BestModel.

**options**

* `--new`: create new BestModel
* `--type mini`: use mini config for testing 
(see `src/chess_zero/configs/mini.py`)

### **Evaluator**

```
python src/chess_zero/run.py eval
```

When executed, Evaluation will start. It evaluates BestModel and the lates next-generation model by playing about 200 games. If next-generation model wins, it becomes BestModel.

**options**

* `--type mini`: use mini config for testing
(see `src/chess_zero/configs/mini.py`)