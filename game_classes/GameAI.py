import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from .Player import Player
from .GameBoard import GameBoard


class GameAI(Player):
    def __init__(
        self,
        symbol,
        input_size=7**2,
        hidden_layer_size=64,
        learning_enabled=False,
        session_dir=None,
    ):
        """
        Initialize the Game AI.
        :param symbol: The symbol representing the AI player on the board.
        :param learning_enabled: Flag to indicate if the AI should learn from gameplay.
        """
        super().__init__(symbol)
        self.board = None
        self.input_size = input_size
        self.model = self.create_model(hidden_layer_size)
        self.learning_enabled = learning_enabled
        self.session_dir = session_dir

    def set_board(self, board):
        self.board = board

    def create_model(self, hidden_layer_size):
        """
        Create the ANN model for the AI.
        """
        model = Sequential(
            [
                Dense(hidden_layer_size, activation="relu", input_shape=(49 + 2,)),
                Dense(
                    self.input_size, activation="softmax"
                ),  # Output layer for move probabilities
            ]
        )
        model.compile(optimizer="adam", loss="categorical_crossentropy")
        return model

    def predict_move(self, board):
        """
        Predict the next move based on the current state of the game board.
        """
        board_input = self.preprocess_board(board)
        print("Board input shape:", board_input.shape)
        move_probabilities = self.model.predict(board_input)[0]
        return self.choose_move(board, move_probabilities)

    def preprocess_board(self, board):
        """
        Convert the board to a suitable format for the ANN.
        """
        # Assuming the board is a 2D array, flatten it and encode symbols
        board = board.board
        encoded_board = []
        for row in board:
            for cell in row:
                encoded_board.append(self.encode_symbol(cell))

        # Reshape the encoded board to match the input shape of the model
        return np.array(encoded_board).reshape(1, -1)

    def choose_move(self, board, move_probabilities):
        valid_moves = self.get_valid_moves(board)
        if not valid_moves:
            return None

        # Convert valid moves to indices
        move_indices = [self.move_to_index(move) for move in valid_moves]

        move_probs = [move_probabilities[idx] for idx in move_indices]

        # Select the move with the highest probability among the valid moves
        best_move_idx = np.argmax(move_probs)

        return valid_moves[best_move_idx]

    def encode_symbol(self, symbol):
        """
        Encode the board symbols for the neural network.
        """
        # Example encoding: 'X' -> 1, 'O' -> -1, ' ' (empty) -> 0
        return 1 if symbol == self.symbol else -1 if symbol != " " else 0

    def move_to_index(self, move):
        # Assuming move is a tuple (source, destination)
        # Only consider the destination for the index
        _, destination = move
        dest_row, dest_col = self.parse_coord(destination)

        # Convert the destination coordinate to a single index
        return dest_row * 7 + dest_col

    def get_valid_moves(self, board):
        """
        Generate a list of valid moves for the AI player using the game board's methods.
        """
        valid_moves = []
        for row in range(board.size):
            for col in range(board.size):
                source = self.convert_coord(row, col)
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    dest_row, dest_col = row + dx, col + dy
                    destination = self.convert_coord(dest_row, dest_col)
                    if board.is_valid_move(source, destination, self.symbol):
                        valid_moves.append((source, destination))
        return valid_moves

    def convert_coord(self, row, col):
        """
        Convert row and column indices to the coordinate format used by the game board.
        """
        # Assuming the game board uses a coordinate system like 'a1', 'b2', etc.
        return f"{chr(97 + row)}{col + 1}"

    def learn_from_game(
        self, game_outcome, reward_win=1, reward_loss=-1, reward_draw=0
    ):
        """
        Update the AI's learning model based on the outcome of a game.
        :param game_outcome: The outcome of the game (win/lose/draw).
        """
        if not self.learning_enabled:
            return

        reward = {"win": reward_win, "lose": reward_loss, "draw": reward_draw}.get(
            game_outcome, 0
        )
        game_data = self.get_game_data()
        X, y = self.prepare_training_data(game_data)
        self.model.fit(X, y, epochs=10)

    def get_game_data(self):
        """
        Get the game data from the game history.
        """
        latest_game_log = self.get_latest_game_log(self.session_dir)
        game_data = []
        unique_moves = []

        with open(latest_game_log, "r") as file:
            for line in file:
                parts = line.strip().split(" - ")
                if len(parts) == 2:
                    timestamp, info = parts
                    if "Unique Moves" in info:
                        # Extract unique moves count
                        unique_moves.append(info)
                    else:
                        game_data.append(info)
        return game_data, unique_moves

    def determine_reward(self, game_outcome_line):
        if "Winner: X" in game_outcome_line:
            return 1 if self.symbol == "X" else -1
        elif "Winner: O" in game_outcome_line:
            return 1 if self.symbol == "O" else -1
        else:
            return 0  # For a draw or ongoing game

    def prepare_training_data(self, game_data, unique_moves):
        X = []  # Feature vectors
        y = []  # Rewards
        current_board_state = [[" " for _ in range(7)] for _ in range(7)]

        # Determine the outcome reward
        game_outcome_line = game_data[-1]  # Last line in game data
        reward = self.determine_reward(game_outcome_line)

        # Process game data and unique moves simultaneously
        for line, unique_move_line in zip(game_data[:-1], unique_moves):
            _, move_info = line.split(" - ")
            _, _, source, _, destination = move_info.split(" ")

            # Process unique moves
            player_unique_moves = self.process_unique_moves(unique_move_line)

            # Update board state and encode it
            self.update_board_state(source, destination, current_board_state)
            encoded_board_state = self.encode_board_state(current_board_state)

            # Include unique moves count in the feature vector
            feature_vector = encoded_board_state + player_unique_moves
            X.append(feature_vector)
            y.append(reward)  # Use the determined reward for each move

        return np.array(X), np.array(y)

    def parse_coord(self, coord):
        row = ord(coord[0]) - ord("a")
        col = int(coord[1]) - 1
        return row, col

    def encode_board_state(self, board_state):
        # Flatten and encode the board state
        encoded_state = []
        for row in board_state:
            for cell in row:
                encoded_state.append(self.encode_symbol(cell))
        return encoded_state

    def get_latest_game_log(self, session_dir):
        """
        Get the path to the latest game log.
        """
        log_files = sorted(
            [
                os.path.join(session_dir, log)
                for log in os.listdir(session_dir)
                if log.startswith("game_log-") and log.endswith(".txt")
            ],
            key=os.path.getmtime,
        )

        return log_files[-1] if log_files else None

    def run_ai_game(self):
        """
        Run a game with AI vs AI and use the outcome to train the AI.
        """

        self.reset_game()
        while not self.is_game_over():
            self.play_ai_turn()

        # Determine the outcome of the game
        outcome = self.check_win_condition()

        # Use the outcome to update AI models
        self.player1.learn_from_game(outcome)
        self.player2.learn_from_game(outcome)
