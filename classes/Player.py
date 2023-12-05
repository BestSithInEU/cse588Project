import numpy as np


class Player:
    """
    The `Player` class represents a player in a game. It has methods to add and move pieces on a game board,
    get valid moves, count unique moves, and reset the player's pieces.
    """

    def __init__(self, symbol: str):
        """
        Initializes a player with the given symbol and an empty set of pieces.

        Parameters
        ----------
        symbol : str
            The symbol representing the player on the game board.
        """
        self.symbol = symbol
        self.pieces = set()

    def parse_coord(self, coord: str) -> tuple[int, int]:
        """
        Parses a coordinate string into row and column indices.

        Parameters
        ----------
        coord : str
            The coordinate string in the format 'a1', 'b2', etc.

        Returns
        -------
        tuple[int, int]
            The row and column indices.
        """
        if len(coord) != 2 or not coord[0].isalpha() or not coord[1:].isdigit():
            raise ValueError("Invalid coordinate format")
        row = ord(coord[0]) - ord("a")
        col = int(coord[1:]) - 1
        return row, col

    def add_piece(self, coord: str) -> None:
        """
        Adds a piece at the given coordinate.

        Parameters
        ----------
        coord : str
            The coordinate string in the format 'a1', 'b2', etc.
        """
        row, col = self.parse_coord(coord)
        if 0 <= row < 8 and 0 <= col < 8:
            self.pieces.add((row, col))

    def move_piece(self, source: str, destination: str) -> None:
        """
        Moves a piece from the source coordinate to the destination coordinate.

        Parameters
        ----------
        source : str
            The source coordinate string in the format 'a1', 'b2', etc.
        destination : str
            The destination coordinate string in the format 'a1', 'b2', etc.
        """
        source_row, source_col = self.parse_coord(source)
        destination_row, destination_col = self.parse_coord(destination)
        if (0 <= source_row < 8 and 0 <= source_col < 8) and (
            0 <= destination_row < 8 and 0 <= destination_col < 8
        ):
            self.pieces.remove((source_row, source_col))
            self.pieces.add((destination_row, destination_col))

    def get_valid_moves(self, board):
        """
        Returns a list of valid moves for the player on the given board.

        Parameters
        ----------
        board : object
            The game board.

        Returns
        -------
        list[tuple[str, str]]
            A list of valid moves in the format [(source, destination), ...].
        """
        valid_moves = []
        for row, col in self.pieces:
            source = self.convert_coord(row, col)
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                dest_row, dest_col = row + dx, col + dy
                if 0 <= dest_row < 7 and 0 <= dest_col < 7:
                    destination = self.convert_coord(dest_row, dest_col)
                    if board.is_valid_move(source, destination, self.symbol):
                        valid_moves.append((source, destination))
        return valid_moves

    def count_unique_moves(self, board) -> int:
        """
        Counts the number of unique destinations for the player's valid moves on the given board.

        Parameters
        ----------
        board : object
            The game board.

        Returns
        -------
        int
            The number of unique destinations.
        """
        unique_destinations = set()
        valid_moves = self.get_valid_moves(board)
        for source, destination in valid_moves:
            unique_destinations.add(destination)
        return len(unique_destinations)

    def reset(self) -> None:
        """
        Resets the player's pieces to an empty set.
        """
        self.pieces = set()

    def convert_coord(self, row: int, col: int) -> str:
        """
        Convert row and column indices to the coordinate format used by the game board.

        Parameters
        ----------
        row : int
            The row index.
        col : int
            The column index.

        Returns
        -------
        str
            The coordinate string in the format 'a1', 'b2', etc.
        """
        if 0 <= row < 7 and 0 <= col < 7:
            return f"{chr(97 + row)}{col + 1}"
        else:
            return ""
