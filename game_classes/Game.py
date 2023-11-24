import os
import logging
import datetime
import random
from .GameBoard import GameBoard
from .GameAI import GameAI
from .Player import Player


class Game:
    session_dir = None

    def __init__(
        self, num_pieces, turn_limit, player1=None, player2=None, random_start=False
    ):
        if num_pieces < 1 or turn_limit < 1:
            raise ValueError("Invalid number of pieces or turn limit")

        self.initial_pieces = num_pieces
        self.turn_limit = turn_limit
        self.play_count = 0
        self.random_start = random_start
        self.move_history = []
        self.undone_moves = []
        self.game_over = False
        self.player1 = player1 or Player("X")
        self.player2 = player2 or Player("O")
        self._setup_game_environment()
        self.reset_game()

    def _setup_game_environment(self):
        self._create_session_directory()
        self._setup_logger()

    def _create_session_directory(self):
        if Game.session_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            Game.session_dir = f"log/session_{timestamp}"
            os.makedirs(Game.session_dir, exist_ok=True)

    def _setup_logger(self):
        log_filename = os.path.join(Game.session_dir, f"game_log-{self.play_count}.txt")
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
        )

    def switch_player(self):
        print(f"Switching player from {self.current_player.symbol}")
        self.current_player = (
            self.player2 if self.current_player == self.player1 else self.player1
        )
        print(f"Current player is now {self.current_player.symbol}")

    def play_ai_turn(self):
        if isinstance(self.current_player, GameAI):
            move = self.current_player.predict_move(self.board)
        if move:
            src_coord, dest_coord = move  # Assuming move is already in the right format
            if self.board.is_valid_move(
                src_coord, dest_coord, self.current_player.symbol
            ):
                self.record_move(src_coord, dest_coord)
                self._execute_move(src_coord, dest_coord)
                self.check_game_status()
            else:
                logging.info(f"AI attempted invalid move: {src_coord} to {dest_coord}")

    def play_turn(self, source, destination):
        # Handle a player's turn
        if not self.current_player.get_valid_moves(self.board):
            return self._end_game_due_to_no_moves()

        src_coord, dest_coord = self._format_coordinates(
            source
        ), self._format_coordinates(destination)

        if self.board.is_valid_move(src_coord, dest_coord, self.current_player.symbol):
            self.record_move(source, destination)
            self._execute_move(src_coord, dest_coord)
            return self.check_game_status()  # Check if the game is over
        else:
            print(f"Invalid move attempted from {source} to {destination}")
            return False

    def play_ai_turn(self):
        if isinstance(self.current_player, GameAI):
            move = self.current_player.predict_move(self.board)

            if move:
                src_coord, dest_coord = move
                if self.board.is_valid_move(
                    src_coord, dest_coord, self.current_player.symbol
                ):
                    self.record_move(src_coord, dest_coord)
                    self._execute_move(src_coord, dest_coord)
                    result = self.check_game_status()

                    return result
                else:
                    logging.info(
                        f"AI attempted invalid move: {src_coord} to {dest_coord}"
                    )

    def record_move(self, source, destination):
        move = {
            "player": self.current_player,
            "source": source,
            "destination": destination,
        }
        self.move_history.append(move)
        self.undone_moves.clear()

    def undo_move(self):
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
        if isinstance(coord, str):
            row = ord(coord[0]) - ord("a")
            col = int(coord[1:]) - 1
            return f"{chr(97 + row)}{col + 1}"
        else:
            return f"{chr(97 + coord[0])}{coord[1] + 1}"

    def _execute_move(self, src_coord, dest_coord):
        self.board.update_board(src_coord, dest_coord, self.current_player.symbol)
        self.current_player.move_piece(src_coord, dest_coord)
        if self.current_player == self.player2:
            self.current_turn += 1
        player_num = "Player 1" if self.current_player == self.player1 else "Player 2"
        logging.info(f"{player_num} played {src_coord} to {dest_coord}")
        logging.info(
            f"Player 1 Unique Moves: {self.player1.count_unique_moves(self.board)}"
        )
        logging.info(
            f"Player 2 Unique Moves: {self.player2.count_unique_moves(self.board)}"
        )

    def check_game_status(self):
        # Check win condition at the end of each player's turn
        if self.current_player == self.player2:  # Check at the end of Player 2's turn
            winner = self.check_win_condition()
            if winner:
                game_over_message = f"Game Over. Winner: {winner}"
                logging.info(game_over_message)
                self.game_over = True
                return "GameOver"
            elif self.current_turn >= self.turn_limit:
                logging.info("Game Over. It's a draw.")
                self.game_over = True
                return "GameOver"

        self.switch_player()
        return True

    def _end_game_due_to_no_moves(self):
        winner = (
            self.player2.symbol
            if self.current_player == self.player1
            else self.player1.symbol
        )
        return f"Game Over. Winner: {winner} (opponent has no moves left)"

    def check_win_condition(self):
        player1_unique_moves = self.player1.count_unique_moves(self.board)
        player2_unique_moves = self.player2.count_unique_moves(self.board)

        if player1_unique_moves > player2_unique_moves:
            return self.player1.symbol
        elif player2_unique_moves > player1_unique_moves:
            return self.player2.symbol
        # If equal unique moves, the game may be a draw or continue
        return None

    def is_game_over(self):
        return (
            self.check_win_condition() is not None
            or self.current_turn >= self.turn_limit
        )

    def reset_game(self):
        self.player1.reset()
        self.player2.reset()
        self.current_player = self.player1
        self.board = GameBoard(self.initial_pieces, self.player1, self.player2)
        self.current_turn = 0
        self.game_moves = []
        self.move_history = []
        self.undone_moves = []
        self.play_count += 1
        self.is_game_over = False
        self.game_over = False
        logging.info(
            f"Player 1 Unique Moves: {self.player1.count_unique_moves(self.board)}"
        )
        logging.info(
            f"Player 2 Unique Moves: {self.player2.count_unique_moves(self.board)}"
        )

        # Set the board for AI players
        if isinstance(self.player1, GameAI):
            self.player1.board = self.board
        if isinstance(self.player2, GameAI):
            self.player2.board = self.board
