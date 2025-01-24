import argparse
import h5py

from keras import models    # Model
from keras import layers    # Conv2D, Dense, Flatten, Input

from dlgo import rl
from dlgo import encoders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--board-size', type=int, default=19
    )
    parser.add_argument(
        '--output-file'
    )
    args = parser.parse_args()

    board_size = args.board_size
    output_file = args.output_file

    encoder = encoders.get_encoder_by_name('simple', board_size)
    
    board_input = layers.Input(shape=encoder.shape(), name='board_input')

    conv1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(board_input)
    conv2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(conv1)
    conv3 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(conv2)

    flat = layers.Flatten()(conv3)
    processed_board = layers.Dense(512)(flat)

    policty_hidden_layer = layers.Dense(512, activation='relu')(processed_board)
    policy_output = layers.Dense(encoder.num_points(), activation='softmax')(policty_hidden_layer)

    value_hidden_layer = layers.Dense(512, activation='relu')(processed_board)
    value_output = layers.Dense(1, activation='tanh')(value_hidden_layer)

    model = models.Model(input=board_input, outputs=[policy_output, value_output])

    new_agent = rl.ACAgent(model, encoder)
    with h5py.File(output_file, 'w') as outf:
        new_agent.serialize(outf)


if __name__ == '__main__':
    main()