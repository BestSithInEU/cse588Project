class Player:
    def __init__(self, symbol):
        self.symbol = symbol
        self.pieces = set()  # You can store piece positions as a set of coordinates

    def add_piece(self, coord):
        # Convert coord from string format to tuple
        row, col = ord(coord[0]) - ord("a"), int(coord[1]) - 1
        self.pieces.add((row, col))

    def move_piece(self, source, destination):
        # Convert source and destination to tuple format for set operations
        src_row, src_col = ord(source[0]) - ord("a"), int(source[1]) - 1
        dest_row, dest_col = ord(destination[0]) - ord("a"), int(destination[1]) - 1
        self.pieces.remove((src_row, src_col))
        self.pieces.add((dest_row, dest_col))

    def get_valid_moves(self, board):
        valid_moves = []
        for row in range(7):
            for col in range(7):
                if board.board[row][col] == self.symbol:
                    # Check each direction (up, down, left, right) for valid moves
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        new_row, new_col = row + dx, col + dy
                        if (
                            0 <= new_row < 7
                            and 0 <= new_col < 7
                            and board.board[new_row][new_col] == " "
                        ):
                            valid_moves.append((row, col, new_row, new_col))
        return valid_moves
