import tensorflow as tf

class ExplorationScheduler(tf.keras.callbacks.Callback):
    def __init__(self, target_object, attribute_name, schedule):
        """
        Args:
            target_object: 속성을 변경할 대상 객체 (예: expreplay)
            attribute_name: 변경할 속성의 이름 (예: 'exploration')
            schedule: (step, value) 쌍의 리스트로, 변경 시점을 정의 (예: [(0, 1), (10, 0.1), (400, 0.01)])
        """
        super().__init__()
        self.target_object = target_object
        self.attribute_name = attribute_name
        self.schedule = schedule
        self.current_value = None

    def on_epoch_begin(self, epoch, logs=None):
        # 선형 보간으로 `exploration` 값을 업데이트
        for (start_epoch, start_value), (end_epoch, end_value) in zip(self.schedule[:-1], self.schedule[1:]):
            if start_epoch <= epoch < end_epoch:
                interp_value = start_value + (end_value - start_value) * ((epoch - start_epoch) / (end_epoch - start_epoch))
                setattr(self.target_object, self.attribute_name, interp_value)
                self.current_value = interp_value
                break
        else:
            # 마지막 구간 이후의 값 유지
            setattr(self.target_object, self.attribute_name, self.schedule[-1][1])
            self.current_value = self.schedule[-1][1]

        print(f"Epoch {epoch}: {self.attribute_name} updated to {self.current_value}")

# 예제 대상 객체 (expreplay)
class ExampleReplay:
    def __init__(self):
        self.exploration = 1.0  # 초기 값

# 예제 사용법
expreplay = ExampleReplay()
schedule = [(0, 1.0), (10, 0.1), (400, 0.01)]  # Epoch별 변경 값
exploration_scheduler = ExplorationScheduler(expreplay, 'exploration', schedule)

# 모델 훈련 시 콜백에 추가
model = tf.keras.models.Sequential([tf.keras.layers.Dense(1)])  # 간단한 모델
model.compile(optimizer='adam', loss='mse')
model.fit(
    x=tf.random.normal((100, 1)),
    y=tf.random.normal((100, 1)),
    epochs=20,
    callbacks=[exploration_scheduler]
)