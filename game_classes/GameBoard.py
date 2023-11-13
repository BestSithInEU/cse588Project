import random


class GameBoard:
    def __init__(self, num_pieces, player1, player2):
        self.board = [[" " for _ in range(7)] for _ in range(7)]
        self.place_pieces(num_pieces, player1, player2)

    def place_pieces(self, num_pieces, player1, player2):
        empty_cells = [(i, j) for i in range(7) for j in range(7)]
        for _ in range(num_pieces):
            # Place a piece for Player 1
            row, col = random.choice(empty_cells)
            self.board[row][col] = player1.symbol
            player1.add_piece(self.convert_coord(row, col))
            empty_cells.remove((row, col))

            # Place a piece for Player 2
            row, col = random.choice(empty_cells)
            self.board[row][col] = player2.symbol
            player2.add_piece(self.convert_coord(row, col))
            empty_cells.remove((row, col))

    def convert_coord(self, row, col):
        # Convert row, col to string format (e.g., "a1")
        return f"{chr(97 + row)}{col + 1}"

    def display_board(self):
        print("   " + "  ".join("1234567"))
        for i, row in enumerate(self.board):
            print(f"{chr(97 + i)} {' | '.join(row)}")
            if i < 6:
                print("  " + "-" * 13)

    def update_board(self, source, destination, player_symbol):
        src_row, src_col = ord(source[0]) - ord("a"), int(source[1]) - 1
        dest_row, dest_col = ord(destination[0]) - ord("a"), int(destination[1]) - 1
        self.board[src_row][src_col] = " "
        self.board[dest_row][dest_col] = player_symbol

    def is_valid_move(self, source, destination, player_symbol):
        src_row, src_col = self.parse_coordinates(source)
        dest_row, dest_col = self.parse_coordinates(destination)
        if not (
            0 <= src_row < 7
            and 0 <= src_col < 7
            and 0 <= dest_row < 7
            and 0 <= dest_col < 7
        ):
            print("Move outside the board")
            return False  # Move outside the board
        if abs(src_row - dest_row) + abs(src_col - dest_col) != 1:
            print("Not moving to a neighbor")
            return False  # Not moving to a neighbor
        if (
            self.board[src_row][src_col] != player_symbol
            or self.board[dest_row][dest_col] != " "
        ):
            print("Invalid source or destination")
            return False  # Invalid source or destination
        print("Valid move")
        return True

    def parse_coordinates(self, coord):
        row = ord(coord[0]) - ord("a")
        col = int(coord[1]) - 1
        return row, col
