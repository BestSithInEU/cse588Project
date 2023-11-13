import os
import logging
import datetime
from .GameBoard import GameBoard
from .Player import Player


class Game:
    session_dir = None  # Class attribute to store the session directory

    def __init__(self, num_pieces, turn_limit):
        self.initial_pieces = num_pieces
        self.turn_limit = turn_limit
        self.play_count = 0
        if Game.session_dir is None:
            # First game instance in this application run
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            Game.session_dir = f"log/session_{timestamp}"
            os.makedirs(Game.session_dir, exist_ok=True)
        self.reset_game()

    def setup_logger(self):
        # Check if the session directory has been created
        if Game.session_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            Game.session_dir = f"log/session_{timestamp}"
            os.makedirs(Game.session_dir, exist_ok=True)

        # Create a log file name based on the play count
        log_filename = os.path.join(Game.session_dir, f"game_log-{self.play_count}.txt")

        # Set up the logger
        self.logger = logging.getLogger(f"GameLogger_{self.play_count}")
        self.logger.setLevel(logging.INFO)

        # Prevent adding multiple handlers to the logger
        if not self.logger.hasHandlers():
            handler = logging.FileHandler(log_filename)
            handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            self.logger.addHandler(handler)

    def switch_player(self):
        self.current_player = (
            self.player2 if self.current_player == self.player1 else self.player1
        )

    def play_turn(self, source, destination):
        # Convert coordinates to string format (e.g., "a1")
        src_coord = f"{chr(97 + source[0])}{source[1] + 1}"
        dest_coord = f"{chr(97 + destination[0])}{destination[1] + 1}"
        if self.board.is_valid_move(src_coord, dest_coord, self.current_player.symbol):
            self.board.update_board(src_coord, dest_coord, self.current_player.symbol)
            self.current_player.move_piece(src_coord, dest_coord)
            self.current_turn += 1
            player_num = (
                "Player 1" if self.current_player == self.player1 else "Player 2"
            )
            print(f"{player_num} played {src_coord} to {dest_coord}")
            self.game_moves.append((self.current_player.symbol, src_coord, dest_coord))
            self.logger.info(f"{player_num} played {src_coord} to {dest_coord}")
            self.switch_player()

            if self.is_game_over():
                self.logger.info("Game Over. It's a draw.")
                return "GameOver"
            else:
                print(f"Turn {self.current_turn}. {self.current_player}'s turn")
            return True
        print("Invalid move!")
        self.logger.info(f"Invalid move: {src_coord} to {dest_coord}")
        return False

    def check_win_condition(self):
        # Implement your win condition logic here
        # Return the winning player or None if no winner
        # -1 for player 1, 1 for player 2 (for now) tie is 0
        pass

    def is_game_over(self):
        # winner = self.check_win_condition()
        # if winner:
        #     self.logger.info(f"Game Over. Winner: {winner}")
        #     return True, winner
        if self.current_turn >= self.turn_limit:
            return True
        # Add other draw conditions if needed
        return False

    def reset_game(self):
        self.play_count += 1
        self.player1 = Player("X")
        self.player2 = Player("O")
        self.board = GameBoard(
            self.initial_pieces, self.player1, self.player2
        )  # Pass players
        self.current_player = self.player1
        self.current_turn = 0
        self.game_moves = []
        self.setup_logger()
        if self.play_count > 1:
            print("Game restarted")
            self.logger.info("Game restarted")
