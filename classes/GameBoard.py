import re
import random


class GameBoard:
    """
    Represents a game board.

    Attributes:
        size (int): The size of the game board.
        board (list): The game board represented as a 2D list.
    """

    def __init__(self, num_pieces, player1, player2, size=7):
        """
        Initializes a GameBoard object.

        Args:
            num_pieces (int): The number of pieces to be placed on the board for each player.
            player1 (Player): The first player object.
            player2 (Player): The second player object.
            size (int, optional): The size of the game board. Defaults to 7.
        """
        self.size = size
        self.board = [[" " for _ in range(self.size)] for _ in range(self.size)]
        self._place_pieces(num_pieces, player1, player2)

    def _place_pieces(self, num_pieces, player1, player2):
        """
        Randomly places the pieces on the board for each player.

        Args:
            num_pieces (int): The number of pieces to be placed on the board for each player.
            player1 (Player): The first player object.
            player2 (Player): The second player object.
        """
        empty_cells = [(i, j) for i in range(self.size) for j in range(self.size)]
        for _ in range(num_pieces):
            for player in [player1, player2]:
                cell = random.choice(empty_cells)
                self.board[cell[0]][cell[1]] = player.symbol
                player.add_piece(self._convert_coord(cell[0], cell[1]))
                empty_cells.remove(cell)

    def _convert_coord(self, row, col):
        """
        Converts row and column indices to coordinate strings.

        Args:
            row (int): The row index.
            col (int): The column index.

        Returns:
            str: The coordinate string.
        """
        return f"{chr(97 + row)}{col + 1}"

    def parse_coordinates(self, coord):
        """
        Parses coordinate strings into row and column indices.

        Args:
            coord (str): The coordinate string.

        Returns:
            tuple: The row and column indices.

        Raises:
            ValueError: If the coordinate format is invalid.
        """
        pattern = r"^([a-z])(\d+)$"
        match = re.match(pattern, coord)
        if match:
            row = ord(match.group(1)) - ord("a")
            col = int(match.group(2)) - 1
            return row, col
        else:
            raise ValueError("Invalid coordinate format")

    def _is_within_board(self, row, col):
        """
        Checks if a given row and column indices are within the board boundaries.

        Args:
            row (int): The row index.
            col (int): The column index.

        Returns:
            bool: True if the indices are within the board boundaries, False otherwise.
        """
        return 0 <= row < self.size and 0 <= col < self.size

    def is_valid_move(self, source, destination, player_symbol):
        """
        Checks if a move from source to destination is valid for a player.

        Args:
            source (str): The source coordinate.
            destination (str): The destination coordinate.
            player_symbol (str): The symbol representing the player.

        Returns:
            bool: True if the move is valid, False otherwise.
        """
        if destination is None:
            return False

        src_row, src_col = self.parse_coordinates(source)
        dest_row, dest_col = self.parse_coordinates(destination)
        if not (
            self._is_within_board(src_row, src_col) and self._is_within_board(dest_row, dest_col)
        ):
            return False
        if abs(src_row - dest_row) + abs(src_col - dest_col) != 1:
            return False
        if self.board[src_row][src_col] != player_symbol or self.board[dest_row][dest_col] != " ":
            return False
        return True

    def display_board(self):
        """
        Displays the current state of the board.
        """
        header = "   " + "  ".join(map(str, range(1, self.size + 1)))
        rows = [f"{chr(97 + i)} {' | '.join(row)}" for i, row in enumerate(self.board)]
        separator = "  " + "-" * (self.size * 4 - 1)
        print(header)
        print(separator.join(rows))

    def update_board(self, source, destination, player_symbol):
        """
        Updates the board after a move.

        Args:
            source (str): The source coordinate.
            destination (str): The destination coordinate.
            player_symbol (str): The symbol representing the player.
        """
        src_row, src_col = self.parse_coordinates(source)
        dest_row, dest_col = self.parse_coordinates(destination)
        self.board[src_row][src_col] = " "
        self.board[dest_row][dest_col] = player_symbol
