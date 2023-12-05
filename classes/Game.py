"""This module contains classes related to the game state, configuration, and main game logic."""
import os
import logging
import datetime
from .GameBoard import GameBoard
from .GameAI import GameAI
from .Player import Player


class GameState:
    def __init__(self, num_pieces, turn_limit, random_start):
        self.initial_pieces = num_pieces
        self.turn_limit = turn_limit
        self.play_count = 0
        self.random_start = random_start
        self.move_history = []
        self.undone_moves = []
        self.game_over = False
        self.current_turn = 0


class GameConfig:
    def __init__(self, num_pieces, turn_limit, random_start=False):
        self.num_pieces = num_pieces
        self.turn_limit = turn_limit
        self.random_start = random_start


class Game:
    def __init__(self, config, player1=None, player2=None):
        self.state = GameState(config.num_pieces, config.turn_limit, config.random_start)
        self.player1 = player1 or Player("X")
        self.player2 = player2 or Player("O")
        self.current_player = self.player1
        self.session_dir = None
        self.game_over = False
        self._setup_game_environment()
        self.reset_game()

    def _setup_game_environment(self):
        """
        Set up the game environment. This includes creating a session directory and setting up logging.
        """
        self._create_session_directory()
        self._setup_logger()

    @staticmethod
    def _get_base_path():
        """
        Returns the base path to the project's root directory.
        This method assumes that the current file is not in the root directory
        and needs to navigate up the directory tree.
        """
        current_file = os.path.abspath(__file__)
        # Navigate up directories based on the depth of this file in the project structure
        # For example, if this file is two levels deep from the root, use os.path.dirname twice
        root_dir = os.path.dirname(
            os.path.dirname(current_file)
        )  # Adjust based on your project structure
        return root_dir

    def _create_session_directory(self):
        """
        Creates a session directory in the 'log' folder of the project's root directory.
        """
        if self.session_dir is None:
            base_path = self._get_base_path()
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.session_dir = os.path.join(base_path, f"log/session_{timestamp}")
            os.makedirs(self.session_dir, exist_ok=True)
        return self.session_dir  # Return the session directory path

    def _setup_logger(self):
        """
        Sets up logging for the game session.
        """
        session_dir = self._create_session_directory()  # Ensure session directory is created
        log_filename = os.path.join(session_dir, f"game_log-{self.state.play_count}.txt")
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
        )

    def switch_player(self):
        """The function `switch_player` switches the current player in a game."""

        print(f"Switching player from {self.current_player.symbol}")
        self.current_player = self.player2 if self.current_player == self.player1 else self.player1
        print(f"Current player is now {self.current_player.symbol}")

    def play_ai_turn(self):
        """
        Handle a turn for an AI player.
        """
        move = None
        if isinstance(self.current_player, GameAI):
            valid_moves = self.current_player.get_valid_moves(self.board)
            move = self.current_player.predict_move(self.board, valid_moves)
        if move:
            src_coord, dest_coord = move
            if self.board.is_valid_move(src_coord, dest_coord, self.current_player.symbol):
                self.record_move(src_coord, dest_coord)
                self._execute_move(src_coord, dest_coord)
                self.check_game_status()
            else:
                logging.info("AI attempted invalid move: %s to %s", src_coord, dest_coord)

    def play_turn(self, source, destination):
        """The function `play_turn` handles a player's turn in a game, checking if the move is valid and
        executing it if so.

        Parameters
        ----------
        source
            The source parameter represents the starting position of a piece on the game board. It is the
        position from where the player wants to move a piece.
        destination
            The `destination` parameter in the `play_turn` method represents the coordinate where the
        player wants to move their piece to on the game board.

        Returns
        -------
            either the result of the `_end_game_due_to_no_moves()` method or the result of the
        `check_game_status()` method.

        """
        # Handle a player's turn
        if not self.current_player.get_valid_moves(self.board):
            return self._end_game_due_to_no_moves()

        src_coord, dest_coord = self._format_coordinates(source), self._format_coordinates(
            destination
        )

        if self.board.is_valid_move(src_coord, dest_coord, self.current_player.symbol):
            self.record_move(source, destination)
            self._execute_move(src_coord, dest_coord)
            return self.check_game_status()
        print(f"Invalid move attempted from {source} to {destination}")
        return False

    def record_move(self, source, destination):
        """The function records a move made by a player in a game and adds it to the move history.

        Parameters
        ----------
        source
            The source parameter represents the starting position or location of the move. It could be a
        coordinate, index, or any other identifier that specifies the source of the move.
        destination
            The "destination" parameter in the "record_move" function represents the location where the
        player wants to move their piece to. It could be a specific position on a game board or any
        other designated location in the game.

        """
        move = {
            "player": self.current_player,
            "source": source,
            "destination": destination,
        }
        self.move_history.append(move)
        self.undone_moves.clear()

    def undo_move(self):
        """The `undo_move` function allows the player to undo their last move in a game by reverting the
        board state, player positions, and current turn.

        Returns
        -------
            the last move that was undone.

        """
        if not self.move_history:
            return None
        last_move = self.move_history.pop()
        self.undone_moves.append(last_move)

        # Assuming last_move["source"] and last_move["destination"] are stored as strings
        src_coord = self._format_coordinates(last_move["source"])
        dest_coord = self._format_coordinates(last_move["destination"])

        # Swap source and destination to revert
        self.board.update_board(dest_coord, src_coord, last_move["player"].symbol)
        last_move["player"].move_piece(dest_coord, src_coord)
        self.switch_player()

        if self.current_player == self.player2:
            self.current_turn -= 1

        return last_move

    def redo_move(self):
        """The `redo_move` function allows a player to redo their previous move in a game by updating the
        board, player positions, and move history.

        Returns
        -------
            The `redo_move` method returns the redone move, which is a dictionary containing information
        about the move that was redone.

        """
        if not self.undone_moves:
            return None

        redone_move = self.undone_moves.pop()
        src_coord = self._format_coordinates(redone_move["source"])
        dest_coord = self._format_coordinates(redone_move["destination"])

        # Execute the move
        self.board.update_board(src_coord, dest_coord, redone_move["player"].symbol)
        redone_move["player"].move_piece(src_coord, dest_coord)
        self.switch_player()

        # Increment or decrement the turn based on which player just moved
        if self.current_player == self.player1:
            self.current_turn += 1

        # Add the redone move back to the move history
        self.move_history.append(redone_move)

        return redone_move

    def _format_coordinates(self, coord):
        """The function `_format_coordinates` converts a coordinate string or tuple into a formatted
        string.

        Parameters
        ----------
        coord
            The `coord` parameter represents the coordinates of a position on a grid. It can be either a
        string or a tuple.

        Returns
        -------
            a formatted string representing the coordinates. If the input `coord` is a string, it converts
        the first character to a row index (subtracting the ASCII value of 'a') and the remaining
        characters to a column index (subtracting 1). If the input `coord` is not a string, it assumes
        it is a tuple or list with two elements representing the row and column

        """
        if isinstance(coord, str):
            row = ord(coord[0]) - ord("a")
            col = int(coord[1:]) - 1
            return f"{chr(97 + row)}{col + 1}"
        return f"{chr(97 + coord[0])}{coord[1] + 1}"

    def _execute_move(self, src_coord, dest_coord):
        """The function executes a move in a game, updates the board and player's piece, logs the move, and
        counts the unique moves for each player.

        Parameters
        ----------
        src_coord
            The `src_coord` parameter represents the source coordinate of the piece that the player wants
        to move. It is the current position of the piece on the board.
        dest_coord
            The `dest_coord` parameter in the `_execute_move` method represents the destination coordinate
        where the player wants to move their piece to. It is the coordinate on the game board where the
        piece will be moved to.

        """
        self.board.update_board(src_coord, dest_coord, self.current_player.symbol)
        self.current_player.move_piece(src_coord, dest_coord)
        if self.current_player == self.player2:
            self.current_turn += 1
        player_num = "Player 1" if self.current_player == self.player1 else "Player 2"
        logging.info("%s played %s to %s", player_num, src_coord, dest_coord)
        logging.info("Player 1 Unique Moves: %s", self.player1.count_unique_moves(self.board))
        logging.info("Player 2 Unique Moves: %s", self.player2.count_unique_moves(self.board))

    def check_game_status(self):
        """The function checks the game status at the end of each player's turn and returns the winner or a
        draw message if the game is over.

        Returns
        -------
            either "GameOver" if there is a winner or a draw, or it is returning True if the game is not
        over yet.

        """
        # Check win condition at the end of each player's turn
        if self.current_player == self.player2:  # Check at the end of Player 2's turn
            winner = self.check_win_condition()
            if winner:
                game_over_message = f"Game Over. Winner: {winner}"
                logging.info(game_over_message)
                self.game_over = True
                return "GameOver"
            if self.current_turn >= self.state.turn_limit:
                logging.info("Game Over. It's a draw.")
                self.game_over = True
                return "GameOver"

        self.switch_player()
        return True

    def _end_game_due_to_no_moves(self):
        """The function returns a string indicating the winner of the game when the opponent has no moves
        left.

        Returns
        -------
            a string that states the game is over and specifies the winner as the opponent who has no moves
        left.

        """
        winner = self.player2.symbol if self.current_player == self.player1 else self.player1.symbol
        return f"Game Over. Winner: {winner} (opponent has no moves left)"

    def check_win_condition(self):
        """The function checks the win condition of a game by comparing the number of unique moves made by
        each player.

        Returns
        -------
            the symbol of the player who has more unique moves on the board. If both players have the same
        number of unique moves, the function returns None, indicating that the game may be a draw or
        continue.

        """
        player1_unique_moves = self.player1.count_unique_moves(self.board)
        player2_unique_moves = self.player2.count_unique_moves(self.board)

        if player1_unique_moves > player2_unique_moves:
            return self.player1.symbol
        if player2_unique_moves > player1_unique_moves:
            return self.player2.symbol
        # If equal unique moves, the game may be a draw or continue
        return None

    def is_game_over(self):
        """The function checks if the game is over by checking if the win condition has been met or if the
        current turn is equal to or greater than the turn limit.

        Returns
        -------
            a boolean value.

        """
        return self.check_win_condition() is not None or self.current_turn >= self.state.turn_limit

    def reset_game(self) -> None:
        """
        Reset the game state.
        """
        # Reset player states
        self.player1.reset()
        self.player2.reset()

        # Set current player
        self.current_player = self.player1

        # Create a new game board
        self.board = GameBoard(self.state.initial_pieces, self.player1, self.player2)

        # Reset turn count and move history
        self.current_turn = 0
        self.game_moves = []
        self.move_history = []
        self.undone_moves = []

        # Increase play count
        self.state.play_count += 1

        # Set game_over flag to False
        self.game_over = False

        # Log player unique moves
        logging.info("Player 1 Unique Moves: %s", self.player1.count_unique_moves(self.board))
        logging.info("Player 1 Unique Moves: %s", self.player2.count_unique_moves(self.board))

    def train_ai(self, num_games):
        """
        Trains the AI player by playing a specified number of games.

        Parameters:
        num_games (int): Number of games to be played for training.
        """
        training_data = []
        for game_index in range(num_games):
            self.reset_game()
            turn_number = 0
            ai_player = self.player1 if isinstance(self.player1, GameAI) else self.player2

            while not self.game_over:
                board_before_move = self.board_to_string(self.board)

                # Play a turn
                self.play_ai_turn()

                board_after_move = self.board_to_string(self.board)

                data = {
                    "turn_number": turn_number,
                    "board_before": board_before_move,
                    "board_after": board_after_move,
                    "unique_moves_player1": self.player1.count_unique_moves(self.board),
                    "unique_moves_player2": self.player2.count_unique_moves(self.board),
                }
                training_data.append(data)

                # Game state for training
                if isinstance(ai_player, GameAI):
                    game_state = ai_player.prepare_input_data(self.board)
                    game_result = self.determine_game_result()
                    last_move = self.move_history[-1] if self.move_history else None

                    if last_move is not None:
                        action = ai_player.move_to_index(
                            (last_move["source"], last_move["destination"])
                        )
                    else:
                        action = None
                    next_state = ai_player.prepare_input_data(self.board)
                    done = self.game_over

                    # Update AI model
                    ai_player.remember(
                        game_state,
                        action,
                        ai_player.calculate_reward(game_result),
                        next_state,
                        done,
                    )
                    ai_player.replay(batch_size=32)
                else:
                    logging.warning(
                        "The AI player is not an instance of GameAI. Skipping AI training steps."
                    )

            print(f"Game {game_index + 1} ended.")
            self.save_training_data(training_data, game_index + 1)
            # Optionally save the model after each game
            # ai_player.save_model(f"game_ai_model_{game_index}.h5")

    def board_to_string(self, board):
        """The function `board_to_string` takes a board object and returns a string representation of the
        board.

        Parameters
        ----------
        board : GameBoard
            The `board` parameter is a 2-dimensional list representing a game board. Each element in the
        list represents a row on the board, and each element within a row represents a cell on the
        board. The elements can be either a string representing the content of the cell (e.g., "X

        Returns
        -------
            a string representation of the board.

        """
        board_str = "  " + " ".join(str(i) for i in range(1, len(board.board[0]) + 1)) + "\n"
        row_label = "A"
        for row in board.board:
            board_str += (
                row_label + " " + " | ".join(cell if cell != " " else "." for cell in row) + "\n"
            )
            row_label = chr(ord(row_label) + 1)
        return board_str.strip()

    def save_training_data(self, training_data, game_number):
        """The function saves training data to a file, including turn number, board before and after move,
        and unique moves for each player.

        Parameters
        ----------
        training_data : list
            The `training_data` parameter is a list of dictionaries. Each dictionary represents a single
        entry of training data and contains the following keys:
        game_number : int
            The `game_number` parameter is an integer that represents the number of the game for which the
        training data is being saved. It is used to generate a unique filename for the training data
        file.

        """
        with open(f"training_data_game_{game_number}.txt", "w", encoding="utf-8") as file:
            for entry in training_data:
                file.write(f"Turn Number: {entry['turn_number']}\n")
                file.write(f"Board Before Move:\n{entry['board_before']}\n")
                file.write(f"Board After Move:\n{entry['board_after']}\n")
                file.write(f"Player 1 Unique Moves: {entry['unique_moves_player1']}\n")
                file.write(f"Player 2 Unique Moves: {entry['unique_moves_player2']}\n")
                file.write("\n---\n\n")
        print(f"Training data collected for game {game_number}: {len(training_data)} entries.")

    def determine_game_result(self):
        """The function determines the result of a game based on the winner or if it is a draw.

        Returns
        -------
            a string indicating the result of the game. If the player1 wins, it returns "win". If there is
        a draw, it returns "draw". Otherwise, it returns "lose".

        """

        # Logic to determine the result of the game
        winner = self.check_win_condition()
        if winner == self.player1.symbol:
            return "win"
        if winner is None:
            return "draw"
        return "lose"
