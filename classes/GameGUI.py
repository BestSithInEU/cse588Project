from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QLabel,
    QMessageBox,
    QGridLayout,
    QWidget,
    QStatusBar,
    QTextEdit,
)
from PyQt5.QtCore import (
    Qt,
    QTimer,
)
from PyQt5.QtGui import QFont, QColor, QPalette
from .Game import Game
from .GameAI import GameAI


class GameGUI(QMainWindow):
    BOARD_SIZE = 7
    FONT_MAIN = QFont("Arial", 20)
    FONT_LABEL = QFont("Arial", 16)
    COLOR_PLAYER1 = "red"
    COLOR_PLAYER2 = "blue"
    COLOR_DEFAULT = "black"

    def __init__(self, game):
        super().__init__()
        self.game = game
        self.selected_piece = None
        self.initUI()

    def initUI(self):
        self.applyDarkTheme()
        self.setWindowTitle("AI Board Game")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.grid_layout = QGridLayout(self.central_widget)
        self.grid_layout.setSpacing(10)
        self.createLabels()
        self.createButtons()
        self.createTurnLabel()
        self.createStatusBar()
        self.createUndoRedoButtons()
        self.createMoveHistoryDisplay()
        self.createUniqueMovesLabels()
        self.update_board()

    def createLabels(self):
        # Create column labels (1 to 7) at the top
        for j in range(1, GameGUI.BOARD_SIZE + 1):
            label = QLabel(str(j))
            label.setAlignment(Qt.AlignCenter)
            self.grid_layout.addWidget(label, 0, j)  # Adjusted position

        # Create row labels (a to g) on the left
        for i in range(1, GameGUI.BOARD_SIZE + 1):
            label = QLabel(chr(96 + i))  # ASCII 'a' is 97
            label.setAlignment(Qt.AlignCenter)
            self.grid_layout.addWidget(label, i, 0)  # Adjusted position

    def createUniqueMovesLabels(self):
        self.unique_moves_label_player1 = QLabel("Player 1 Unique Moves: 0")
        self.unique_moves_label_player1.setAlignment(Qt.AlignCenter)
        self.unique_moves_label_player1.setFont(GameGUI.FONT_LABEL)
        self.unique_moves_label_player2 = QLabel("Player 2 Unique Moves: 0")
        self.unique_moves_label_player2.setAlignment(Qt.AlignCenter)
        self.unique_moves_label_player2.setFont(GameGUI.FONT_LABEL)
        self.grid_layout.addWidget(self.unique_moves_label_player1, 11, 0, 1, 3)
        self.grid_layout.addWidget(self.unique_moves_label_player2, 11, 4, 1, 3)
        self.update_unique_moves_labels()

    def update_unique_moves_labels(self):
        player1_unique_moves = self.game.player1.count_unique_moves(self.game.board)
        player2_unique_moves = self.game.player2.count_unique_moves(self.game.board)
        self.unique_moves_label_player1.setText(
            f"Player 1 Unique Moves: {player1_unique_moves}"
        )
        self.unique_moves_label_player2.setText(
            f"Player 2 Unique Moves: {player2_unique_moves}"
        )

    def createUndoRedoButtons(self):
        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo_move)
        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(self.redo_move)
        self.grid_layout.addWidget(self.undo_button, 9, 0)
        self.grid_layout.addWidget(self.redo_button, 9, 1)

    def createMoveHistoryDisplay(self):
        self.move_history_display = QTextEdit()
        self.move_history_display.setReadOnly(True)
        self.grid_layout.addWidget(self.move_history_display, 10, 0, 1, 7)

    def undo_move(self):
        last_move = self.game.undo_move()
        if last_move:
            print("Undo move:", last_move)
            self.update_board()
            self.update_turn_label()
            self.update_move_history_display()
            self.update_unique_moves_labels()
        else:
            print("No move to undo")

    def redo_move(self):
        redone_move = self.game.redo_move()
        if redone_move:
            print("Redo move:", redone_move)
            self.update_board()
            self.update_turn_label()
            self.update_move_history_display()
            self.update_unique_moves_labels()
        else:
            print("No move to redo")

    def update_turn_label(self):
        if self.game.current_player == self.game.player1:
            player_text = "<span style='color: red;'>Player 1 (X)</span>"
        else:
            player_text = "<span style='color: blue;'>Player 2 (O)</span>"
        current_turn_text = (
            f"Turn {self.game.current_turn}/{self.game.turn_limit}: {player_text}"
        )
        self.turn_label.setText(current_turn_text)

    def update_move_history_display(self):
        history_text = "\n".join(
            [self.format_move(move) for move in self.game.move_history]
        )
        self.move_history_display.setText(history_text)

    def format_move(self, move):
        # Convert coordinates to 'letter-number' format
        def format_coord(coord):
            if isinstance(coord, str):
                return coord  # Already in the right format
            else:
                # Convert tuple to 'a1', 'b2', etc.
                row, col = coord
                return f"{chr(97 + row)}{col + 1}"

        source_formatted = format_coord(move["source"])
        destination_formatted = format_coord(move["destination"])
        return f"Player {move['player'].symbol} moved from {source_formatted} to {destination_formatted}"

    def applyDarkTheme(self):
        darkPalette = QPalette()
        darkPalette.setColor(QPalette.Window, QColor(53, 53, 53))
        darkPalette.setColor(QPalette.WindowText, Qt.white)
        darkPalette.setColor(QPalette.Base, QColor(25, 25, 25))
        darkPalette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        darkPalette.setColor(QPalette.ToolTipBase, Qt.white)
        darkPalette.setColor(QPalette.ToolTipText, Qt.white)
        darkPalette.setColor(QPalette.Text, Qt.white)
        darkPalette.setColor(QPalette.Button, QColor(53, 53, 53))
        darkPalette.setColor(QPalette.ButtonText, Qt.white)
        darkPalette.setColor(QPalette.BrightText, Qt.red)
        darkPalette.setColor(QPalette.Link, QColor(42, 130, 218))
        darkPalette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        darkPalette.setColor(QPalette.HighlightedText, Qt.black)
        QApplication.setPalette(darkPalette)

    def createButtons(self):
        self.buttons = [
            [self.createButton(row, col) for col in range(1, GameGUI.BOARD_SIZE + 1)]
            for row in range(1, GameGUI.BOARD_SIZE + 1)
        ]

    def createButton(self, row, col):
        button = QPushButton(" ")
        button.setFont(GameGUI.FONT_MAIN)
        button.setStyleSheet(
            "QPushButton { color: white; } QPushButton:hover { background-color: #505050; }"
        )
        button.clicked.connect(lambda state, x=row, y=col: self.on_button_click(x, y))
        self.grid_layout.addWidget(button, row, col)
        return button

    def createTurnLabel(self):
        self.turn_label = QLabel("Player 1's Turn")
        self.turn_label.setAlignment(Qt.AlignCenter)
        self.turn_label.setFont(GameGUI.FONT_LABEL)
        self.turn_label.setStyleSheet("color: white;")
        self.grid_layout.addWidget(self.turn_label, 8, 0, 1, 7)

    def createStatusBar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.setStyleSheet("color: white;")

    def on_button_click(self, row, col):
        if self.game.game_over:
            print("Game is over, no more moves allowed.")
            return

        row -= 1  # Adjusted position
        col -= 1  # Adjusted position
        print(f"Button clicked at {row}, {col}")

        if not self.selected_piece:
            if self.is_valid_selection(row, col):
                # Select the piece
                self.selected_piece = (row, col)
        else:
            # Perform the move
            source = self.selected_piece
            destination = (row, col)
            result = self.game.play_turn(source, destination)
            self.selected_piece = None  # Reset the selected piece
            if result == "GameOver":
                self.handle_game_over()
            elif result:
                self.update_board()
                self.update_unique_moves_labels()

    def is_valid_selection(self, row, col):
        piece = self.game.board.board[row][col]
        return piece == self.game.current_player.symbol

    def update_board(self):
        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                button = self.buttons[i][j]
                symbol = self.game.board.board[i][j]
                button.setText(symbol)
                if symbol == "X":
                    button.setStyleSheet("color: red;")
                elif symbol == "O":
                    button.setStyleSheet("color: blue;")
                else:
                    button.setStyleSheet("color: black;")
        self.update_turn_label()
        self.update_move_history_display()
        print("Updating board...")
        if not self.game.game_over and isinstance(self.game.current_player, GameAI):
            QTimer.singleShot(500, self.trigger_ai_move)

    def trigger_ai_move(self):
        # Check if it's still the AI's turn and the game is not over
        print(self.game.current_player)
        if not self.game.game_over and isinstance(self.game.current_player, GameAI):
            # Play the AI turn and then check again
            self.game.play_ai_turn()
            if not self.game.game_over and isinstance(self.game.current_player, GameAI):
                QTimer.singleShot(500, self.trigger_ai_move)
            else:
                # It's no longer the AI's turn, update the board and UI accordingly
                self.update_board()

    def handle_game_over(self):
        self.update_board()  # Ensure the final board state is shown
        # Determine the winner or if it's a draw
        winner = self.game.check_win_condition()
        if winner:
            winner_message = f"Game Over. Winner: {winner}"
        else:
            winner_message = "Game Over. It's a draw. Would you like to play again?"

        # Display a message box with the game outcome
        message_box = QMessageBox()
        message_box.setWindowTitle("Game Over")
        message_box.setText(winner_message)
        message_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        if message_box.exec_() == QMessageBox.Yes:
            self.reset_game()
        else:
            self.exit_game()

    def reset_game(self):
        print("Resetting game...")
        self.game.reset_game()
        for row_buttons in self.buttons:
            for button in row_buttons:
                button.setText(" ")
                button.setStyleSheet("color: black;")
        self.update_board()

    def exit_game(self):
        # Logic to close the application
        self.close()
