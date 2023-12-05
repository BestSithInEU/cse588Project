from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import random
from collections import deque
from .classes.Player import Player
from keras.utils import to_categorical
from keras.losses import MeanSquaredError


class GameAI(Player):
    def __init__(
        self,
        symbol,
        learning_enabled=False,
        input_size=51,
        hidden_layer_size=64,
        maxlen=2000,
        gamma=0.95,
        learning_rate=0.001,
        learning_rate_decay=0.99,
    ):
        super().__init__(symbol)
        self.learning_enabled = learning_enabled
        self.game = None
        self.memory = deque(maxlen=maxlen)
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.model = self.create_model(input_size, hidden_layer_size)

    def set_game(self, game):
        self.game = game

    def create_model(self, input_size, hidden_layer_size):
        model = Sequential(
            [
                Dense(hidden_layer_size, activation="relu", input_shape=(input_size,)),
                Dense(hidden_layer_size, activation="relu"),
                Dense(49, activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.model.predict(next_state)[0]
                )
            target_f = self.model.predict(state)
            # target_f[0][self.move_to_index(action)] = target
            # self.model.fit(state, target_f, epochs=1, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        self.learning_rate *= self.learning_rate_decay

    def predict_move(self, board, valid_moves):
        if not valid_moves:
            return None

        game_state = self.prepare_input_data(board)
        game_state = game_state.astype(np.float32)
        prediction = self.model.predict(game_state)[0]

        best_move = self.choose_best_valid_move(prediction, valid_moves)
        return best_move

    def choose_best_valid_move(self, prediction, valid_moves):
        move_scores = {
            move: prediction[self.move_to_index(move)] for move in valid_moves
        }

        best_move = max(move_scores, key=move_scores.get)
        return best_move

    def move_to_index(self, move):
        _, dest_coord = move
        dest_index = self.coord_to_index(dest_coord)
        return dest_index

    def coord_to_index(self, coord):
        row = ord(coord[0]) - ord("a")
        col = int(coord[1:]) - 1
        return row * 7 + col

    def prepare_input_data(self, board):
        flattened_board = [
            1.0 if cell == self.symbol else -1.0 if cell != " " else 0.0
            for row in board.board
            for cell in row
        ]

        unique_moves_player1 = self.game.player1.count_unique_moves(board)
        unique_moves_player2 = self.game.player2.count_unique_moves(board)

        input_data = flattened_board + [
            float(unique_moves_player1),
            float(unique_moves_player2),
        ]
        return np.array([input_data])

    def convert_index_to_move(self, index):
        # Convert a move index back to a board coordinate
        row = index // 7
        col = index % 7
        return self.convert_coord(row, col)

    def train_model(self, training_data):
        if self.learning_enabled:
            # Example: Separate features and labels
            X = np.array([data["input"] for data in training_data])
            y = np.array([data["label"] for data in training_data])

            # Convert labels to categorical
            y = to_categorical(y, num_classes=49)

            # Train the model
            self.model.fit(X, y, epochs=10, batch_size=32)

    def update_model(self, batch_size=32):
        if not self.learning_enabled or len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.model.predict(next_state)[0]
                )

            target_f = self.model.predict(state)
            target_f[0][self.action_to_index(action)] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        self.learning_rate *= self.learning_rate_decay
        self.adjust_model_learning_rate()

    def adjust_model_learning_rate(self):
        # Adjust the learning rate of the model's optimizer
        K.set_value(self.model.optimizer.learning_rate, self.learning_rate)

    def save_model(self, filepath):
        self.model.save(filepath)
