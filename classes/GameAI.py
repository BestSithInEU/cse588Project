import json
import random
from collections import deque

import numpy as np
from keras import backend as K
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

from .Player import Player


class GameAI(Player):
    """The `GameAI` class represents an AI player in a game, with the ability to learn from experiences and
    make predictions using a neural network model."""

    def __init__(
        self,
        symbol,
        learning_enabled=False,
        input_size=51,
        hidden_layer_size=64,
    ):
        """The function initializes a class object with various attributes and
        creates a neural network model.

        Parameters
        ----------
        symbol
            The "symbol" parameter is used to represent the symbol or character that the agent will be
        playing as in the game. It could be any symbol or character, such as 'X' or 'O' in a tic-tac-toe
        game.
        learning_enabled, optional
            A boolean value indicating whether the learning process is enabled or not. If set to True, the
        agent will learn from its experiences and update its model. If set to False, the agent will only
        use its current model to make decisions without learning.
        input_size, optional
            The input size is the number of features or inputs that the model will receive. In this case,
        it is set to 51.
        hidden_layer_size, optional
            The `hidden_layer_size` parameter represents the number of neurons in the hidden layer of the
        neural network model. It determines the complexity and capacity of the model to learn and
        represent patterns in the input data. A larger hidden layer size can potentially allow the model
        to learn more complex patterns but may also increase
        """
        super().__init__(symbol)
        self.learning_enabled = learning_enabled
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.game = None
        self.memory = deque(maxlen=2000)  # Experience replay memory
        self.gamma = 0.95  # Discount factor for Temporal Difference Learning
        self.learning_rate = 0.001  # Initial learning rate
        self.learning_rate_decay = 0.999  # Learning rate decay
        self.model = self.create_model(self.input_size, self.hidden_layer_size)

    def create_model(self, input_size, hidden_layer_size):
        """The function creates a sequential model with multiple hidden layers
        and returns the compiled model.

        Parameters
        ----------
        input_size
            The input size is the number of features or dimensions in the input data. It represents the
        number of input neurons in the first layer of the neural network.
        hidden_layer_size
            The `hidden_layer_size` parameter represents the number of neurons in each hidden layer of the
        model. It determines the capacity of the model to learn complex patterns and relationships in
        the data. Increasing the `hidden_layer_size` can potentially improve the model's ability to
        capture intricate patterns but may also increase the

        Returns
        -------
            a compiled Keras model.
        """

        model = Sequential(
            [
                Dense(hidden_layer_size, input_shape=(input_size,)),
                Activation("relu"),
                Dropout(0.3),
                Dense(hidden_layer_size),
                Activation("relu"),
                Dropout(0.3),
                Dense(hidden_layer_size),
                Activation("relu"),
                Dense(49, activation="softmax"),
            ]
        )
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["accuracy"])
        return model

    def set_hyperparameters(self, hyperparams):
        """The function sets the hyperparameters for a model and rebuilds the
        model with the new hyperparameters.

        Parameters
        ----------
        hyperparams
            The `hyperparams` parameter is a dictionary that contains the hyperparameters for the model. It
        should have the following keys:
        """

        self.gamma = hyperparams["gamma"]
        self.learning_rate = hyperparams["learning_rate"]
        self.learning_rate_decay = hyperparams["learning_rate_decay"]
        # Rebuild model with new hyperparameters
        self.model = self.create_model(self.input_size, self.hidden_layer_size)

    def set_game(self, game):
        """The function sets the value of the "game" attribute for an object.

        Parameters
        ----------
        game
            The "game" parameter is the name or identifier of the game that you want to set for the object.

        """
        self.game = game

    def predict_move(self, board, valid_moves):
        """The function predicts the best move to make based on the current
        board state and a list of valid moves.

        Parameters
        ----------
        board
            The "board" parameter represents the current state of the game board. It could be a 2D array or
        any other data structure that represents the positions of the game pieces.
        valid_moves
            The parameter "valid_moves" is a list of all the possible moves that can be made on the current
        board state.

        Returns
        -------
            the best move to make based on the given board and valid moves.
        """
        if not valid_moves:
            return None
        game_state = self.prepare_input_data(board)
        game_state = game_state.astype(np.float32)
        prediction = self.model.predict(game_state)[0]
        best_move = self.choose_best_valid_move(prediction, valid_moves)
        return best_move

    def choose_best_valid_move(self, prediction, valid_moves):
        """The function "choose_best_valid_move" takes a prediction and a list
        of valid moves, and returns the move with the highest score according
        to the prediction.

        Parameters
        ----------
        prediction
            The `prediction` parameter is a dictionary that contains scores for each possible move. The
        keys of the dictionary are the moves, and the values are the corresponding scores.
        valid_moves
            The parameter "valid_moves" is a list of possible moves that can be made in a game.

        Returns
        -------
            the best valid move based on the prediction scores.
        """
        move_scores = {move: prediction[self.move_to_index(move)] for move in valid_moves}
        if move_scores:
            best_move = max(
                (score for score in move_scores.items() if score[1] is not None),
                key=lambda x: x[1],
                default=None,
            )
        else:
            best_move = None
        return best_move

    def move_to_index(self, move):
        """The function "move_to_index" takes a move as input and returns the
        index of the destination coordinate.

        Parameters
        ----------
        move
            A tuple containing two elements. The first element is not used in this function, and the second
        element is the destination coordinate.

        Returns
        -------
            the index corresponding to the destination coordinate of the move.
        """
        _, dest_coord = move
        dest_index = self.coord_to_index(dest_coord)
        return dest_index

    def coord_to_index(self, coord):
        """Convert a coordinate in the form of a string to an index in a 2D
        array.

        Parameters
        ----------
        coord : str
            A string representing a coordinate on a grid. It consists of a letter
            representing the row (from "a" to "g") and a number representing the
            column (from 1 to 7).

        Returns
        -------
        int
            The index corresponding to the given coordinate.
        """
        row = ord(coord[0]) - ord("a")
        col = int(coord[1:]) - 1
        return row * 7 + col

    def prepare_input_data(self, board):
        """The function prepares the input data for a neural network by
        converting the game board into a flattened array and adding additional
        information about the number of unique moves for each player.

        Parameters
        ----------
        board : GameBoard
            The "board" parameter is a representation of the game board. It is expected to be a
        2-dimensional list or array, where each element represents a cell on the board. The elements can
        be either the symbol of a player (e.g., "X" or "O") or an empty

        Returns
        -------
            a numpy array containing the input data.
        """
        flattened_board = [
            1.0 if cell == self.symbol else -1.0 if cell != " " else 0.0
            for row in board.board
            for cell in row
        ]
        if self.game:
            unique_moves_player1 = self.game.player1.count_unique_moves(board)
            unique_moves_player2 = self.game.player2.count_unique_moves(board)
        else:
            unique_moves_player1 = unique_moves_player2 = 0
        input_data = flattened_board + [
            float(unique_moves_player1),
            float(unique_moves_player2),
        ]
        return np.array([input_data])

    def remember(self, state, action, reward, next_state, done):
        """The function appends a tuple of state, action, reward, next_state,
        and done to a memory list.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment. It represents the current situation or configuration of
        the system at a given time.
        action : int
            The action parameter represents the action taken by the agent in the current state.
        reward : float
            The reward parameter represents the reward received by the agent for taking the specified
        action in the given state. It is a numerical value that indicates the desirability or quality of
        the action taken.
        next_state : np.ndarray
            The next state is the state that the environment transitions to after the agent takes an action
        in the current state. It represents the state that the agent will observe next.
        done : bool
            The "done" parameter is a boolean value that indicates whether the episode has ended or not. It
        is typically used in reinforcement learning algorithms to determine when to stop the current
        episode and start a new one.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """The replay function replays and learns from a batch of remembered
        actions.

        Parameters
        ----------
        batch_size : int
            The `batch_size` parameter determines the number of actions to replay and learn from in each
        iteration of the replay process. It specifies how many samples from the memory should be
        randomly selected for training the model.

        Returns
        -------
            If the length of the memory is less than the batch size, nothing is returned. Otherwise, the
        function replays the remembered actions and learns from them.
        """
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose="0")
        self.learning_rate *= self.learning_rate_decay
        self.adjust_model_learning_rate()

    def adjust_model_learning_rate(self):
        """The function adjusts the learning rate of a model's optimizer."""

        K.set_value(self.model.optimizer.lr, self.learning_rate)

    def update_model(self, batch_size=32):
        """The `update_model` function updates the model by storing the current
        state, action, reward, next state, and whether the game is over, and
        then replays a batch of previous experiences.

        Parameters
        ----------
        batch_size : int, optional
            The batch_size parameter determines the number of samples that will be used in each training
        iteration. It specifies how many samples will be fed into the model at once for updating the
        weights.

        Returns
        -------
            If `self.learning_enabled` is `False`, then nothing is returned. Otherwise, if `self.game` is
        `None`, a warning message is printed and nothing is returned.
        """

        if not self.learning_enabled:
            return

        if self.game:
            last_move = self.game.move_history[-1]
            game_result = self.game.determine_game_result()

            action = self.move_to_index((last_move["source"], last_move["destination"]))
            next_state = self.prepare_input_data(self.game.board)
            done = self.game.game_over
            current_state = self.prepare_input_data(last_move["board_before"])
            reward = self.calculate_reward(game_result)

            self.remember(current_state, action, reward, next_state, done)
            self.replay(batch_size)
        else:
            print("Warning: self.game is None. Cannot update model.")

    def replay_minibatch(self, batch_size):
        """Replay a minibatch of the remembered actions and learn from them.

        Parameters:
        batch_size (int): The size of the minibatch to replay.

        """
        if len(self.memory) < batch_size:
            return
        minibatch_generator = self.generate_minibatches(batch_size)
        for minibatch in minibatch_generator:
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
                target_f = self.model.predict(state)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose="0")
        self.learning_rate *= self.learning_rate_decay
        self.adjust_model_learning_rate()

    def generate_minibatches(self, batch_size):
        """Generate minibatches of the remembered actions.

        Parameters:
        batch_size (int): The size of the minibatches to generate.

        Yields:
        list: A minibatch of remembered actions.

        """
        minibatch = random.sample(self.memory, batch_size)
        yield minibatch

    def calculate_reward(self, game_result):
        """Calculate the reward for a game result.

        Parameters:
        game_result (str): The result of the game ('win', 'lose', or 'draw').

        Returns:
        int: The reward for the game result.

        """
        if game_result == "win":
            return 1
        elif game_result == "lose":
            return -1
        return 0  # Draw

    def save_hyperparameters(self, filepath):
        """Save the hyperparameters to a file.

        Parameters:
        filepath (str): The path to the file where the hyperparameters will be saved.

        """
        hyperparams = {
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "learning_rate_decay": self.learning_rate_decay,
        }
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(hyperparams, file)

    def load_hyperparameters(self, filepath):
        """Load the hyperparameters from a file.

        Parameters:
        filepath (str): The path to the file where the hyperparameters will be loaded from.

        """
        with open(filepath, "r", encoding="utf-8") as file:
            hyperparams = json.load(file)
        self.set_hyperparameters(hyperparams)

    def save_model(self, filepath):
        """Save the model to a file.

        Parameters:
        filepath (str): The path to the file where the model will be saved.

        """
        self.model.save(filepath)
