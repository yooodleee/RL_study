import abc
import tensorflow as tf

from tensorpack import ModelDesc
# from tensorpack.compat import tfv1 as tf  -> TensorFlow 1.x 스타일 코드(unit8 x)
'''
TensorFlow 1.x:
    tf.Session()을 사용하여 모델을 실행하고, tf.Graph()를 통해 연산을 정의합니다.
TensorFlow 2.x:
    **즉시 실행(Eager Execution)**이 기본 활성화되어, 세션이나 그래프를 명시적으로 관리하지 않아도 됩니다.

TensorFlow 1.x:
    tf.placeholder와 tf.Session.run()을 사용하여 변수를 정의하고 데이터를 삽입했지만, 
TensorFlow 2.x:
    **tf.keras**를 사용하여 모델을 만들고 훈련하는 방식이 더 일반적입니다.
        Placeholder 제거:
            TensorFlow 2.x에서는 tf.placeholder를 사용하지 않고, 대신 tf.keras.Input()이나 tf.data.Dataset을 사용하여 데이터를 다룹니다.
        세션 제거:
            TensorFlow 2.x에서는 Eager Execution이 기본적으로 활성화되어, 더 이상 세션을 명시적으로 만들지 않아도 됩니다.
        데이터 타입 정의와 변수 선언:
            TensorFlow 2.x에서는 tf.uint8과 같은 타입을 사용하며, tf.Variable을 사용하여 변수를 선언할 수 있습니다.
'''
from tensorpack.tfutils import gradproc, optimizer, summary, varreplace
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.utils import logger


class Model(ModelDesc):

    # TensorFlow 2.x에서는 tf.compat.v1을 사용할 필요 없음
    # tf.keras.Input을 사용하여 입력 텐서를 정의합니다. TensorFlow 2.x에서 placeholder는 더 이상 필요하지 않습니다. 
    # 대신 Input을 사용하여 모델의 입력을 정의하고, 이를 tf.keras.Model에 연결합니다.
    state_dtype=tf.uint8
    state=tf.keras.Input(shape=(64, 64, 3), dtype=state_dtype)  # placeholder 대신 Input 사용

    # reward discount factor
    gamma=0.99

    def __init__(self, state_shape, history, method, num_actions):
        """
        Args:
            state_shape (tuple[int]),
            history (int):
        """
        self.state_shape=tuple(state_shape)
        self._stacked_state_shape= (-1, ) + self.state_shape + (history, )
        self.history=history
        self.method=method
        self.num_actions=num_actions

    def inputs(self):
        # When we use h history frames, the current state and the next state will have (h-1) overlapping frames.
        # Therefore we use a combined state for efficicency:
        # The first h are the current state, and the last h are the next state.
        return [tf.TensorSpec((None,) + self.state_shape (self.history + 1, ), self.state_dtype, 'comb_state'),
                tf.TensorSpec((None,), tf.int64, 'action'),
                tf.TensorSpec((None,), tf.float32, 'reward'),
                tf.TensorSpec((None,), tf.bool, 'isOver')]
                # tf.TensorSpec은 텐서의 shape, dtype, 그리고 이름을 지정하는 데 사용됩니다.
                # None은 배치 크기를 나타내며, 동적으로 크기가 변할 수 있다는 의미입니다.
    
    @abc.abstractmethod
    def _get_DQN_prediction(self, state):
        """
        state: N + state_shape + history
        """
        pass

    @auto_reuse_variable_scope
    def get_DQN_prediction(self, state):
        return self._get_DQN_prediction(state)
    
    def build_graph(self, comb_state, action, reward, isOver):
        comb_state=tf.cast(comb_state, tf.float32)
        input_rank=comb_state.shape.rank

        state=tf.slice(
            comb_state,
            [0] * input_rank,
            [-1] * (input_rank - 1) + [self.history], name='state')
        
        self.predict_value=self.get_DQN_prediction(state)
        if not self.training:
            return
        
        reward=tf.clip_by_value(reward, -1, 1)
        next_state=tf.slice(
            comb_state,
            [0] * (input_rank - 1) + [1],
            [-1] * (input_rank - 1) + [self.history], name='next_state')
        next_state=tf.reshape(next_state, self._stacked_state_shape)
        action_onehot=tf.one_hot(action, self.num_actions, 1.0, 0.0)

        pred_action_value=tf.reduce_sum(self.predict_value * action_onehot, 1)  # N,
        max_pred_reward=tf.reduce_mean(tf.reduce_max(
            self.predict_value, 1), name='predict_reward')
        summary.add_moving_summary(max_pred_reward)

        # tf.variable_scope 제거: TensorFlow 2.x에서는 tf.variable_scope가 더 이상 사용되지 않으며,
        # 대신 변수는 tf.keras.layers나 tf.Module을 통해 관리됩니다. 
        # 변수를 정의할 때 Keras 모델을 사용하거나 tf.Module을 사용합니다.(variable_scope -> Module)

        # varreplace.freeze_variables 제거: 
        # TensorFlow 2.x에서 freeze_variables는 tf.keras.layers.Layer의 파라미터를 trainable=False로 설정하여 
        # 변수를 동결하는 방식으로 대체됩니다.

        # 변수 관리: TensorFlow 2.x에서는 변수(Weights)가 자동으로 관리되기 때문에 별도로 변수 범위를 설정할 필요는 없습니다.
        with tf.Module('target'), varreplace.freeze_variables(skip_collection=True):
            targetQ_predict_value=self.get_DQN_prediction(next_state)   # NxA
        
        if self.method != 'Double':
            # DQN
            best_v=tf.reduce_max(targetQ_predict_value, axis=1)  # N,
            # tf.reduce_max(targetQ_predict_value, axis=1)은 targetQ_predict_value 텐서에서 
            # 각 배치에 대해 두 번째 차원(axis=1)을 따라 최댓값을 계산합니다.
            # 결과는 [batch_size] 모양의 텐서가 됩니다. 즉, 각 샘플마다 최대값을 얻습니다.
            # 1 대신 axis=1로 명시하는 것이 좋은 습관이다.
        else:
            # Double-DQN
            next_predict_value=self.get_DQN_prediction(next_state)
            self.greedy_choices=tf.argmax(next_predict_value, 1)    # N,
            predict_onehot=tf.one_hot(self.greedy_choices, self.num_actions, 1.0, 0.0)
            best_v=tf.reduce_sum(targetQ_predict_value * predict_onehot, 1)
        
        target=reward + (1.0 - tf.cast(isOver, tf.float32)) * self.gamma * tf.stop_gradient(best_v)

        # tf.losses.huber_loss는 tf.keras.losses.Huber로 변경
        cost=tf.keras.losses.Huber(
            # tf.losses.Reduction에서 MEAN은 tf.keras.losses.Reduction으로 변경
            target, pred_action_value, reduction=tf.keras.losses.Reduction.MEAN)
        summary.add_param_summary(('conv.*/W', ['histogram', 'rams']),
                                  ('fc.*/W', ['histogram', 'rms'])) # monitor all W
        summary.add_moving_summary(cost)
        return cost
    
    def optimizer(self):
        # tf.get_variable은 tf.compat.v1.get_variable 또는 **tf.Variable**로 대체
        lr=tf.Variable('learning_rate', initializer=1e-3, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)
        # tf.train.RMSPropOptimizer를 사용하던 코드를 tf.keras.optimizers.RMSprop으로 수정
        opt=tf.keras.optimizers.RMSprop(lr, decay=0.95, momentum=0.95, epsilon=1e-2)
        return optimizer.apply_grad_processors(opt, [gradproc.SummaryGradient()])
    
    @staticmethod
    def update_target_param():
        # tf.global_variables()를 사용할 경우 tf.compat.v1.global_variables()로 대체
        # TensorFlow 2.x에서 TensorFlow 1.x 방식으로 global variables 사용
        # Eager execution 비활성화
        vars=tf.compat.v1.global_variables()
        vars_mapping={x.name: x for x in vars}
        ops=[]
        for v in vars:
            target_name=v.name
            if target_name.startswith('target'):
                new_name=target_name.replace('target/', '')
                logger.info("Target Network Update: {} <- {}".format(target_name, new_name))
                ops.append(v.assign(vars_mapping[new_name]))
        return tf.group(*ops, name='update_target_network')