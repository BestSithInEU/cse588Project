from PyQt5.QtWidgets import (
    QApplication,
    QInputDialog,
)
from PyQt5.QtGui import (
    QPalette,
    QColor,
)
from PyQt5.QtCore import Qt
from game_classes import (
    Game,
    GameGUI,
    GameAI,
    Player,
)


def get_input(prompt: str, valid_options: list) -> str:
    """
    Displays a dialog box to the user, prompting them to select an item from a list of valid options.
    Returns the selected choice.

    Args:
        prompt (str): The message displayed in the dialog box to prompt the user for input.
        valid_options (list): The list of valid options that the user can choose from.

    Returns:
        str: The selected choice from the dialog box.
    """
    try:
        options_text = valid_options
        choice, _ = QInputDialog.getItem(
            None, "Input", prompt, options_text, editable=False
        )
        return choice
    except TypeError:
        raise ValueError("valid_options must contain only strings")


def apply_dark_theme(app):
    darkPalette = QPalette()

    # Set color for different aspects of the UI
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

    # Set the modified palette to the QApplication instance
    app.setPalette(darkPalette)


def play_player_vs_player_game(initial_pieces, turn_limit, app):
    game = Game(initial_pieces, turn_limit)
    app = QApplication([])
    gui = GameGUI(game)
    gui.show()
    app.exec_()


def train_ai_vs_ai(initial_pieces, turn_limit, num_games=100):
    """
    Trains two AI players against each other.

    Args:
        initial_pieces (int): The number of pieces each player starts with.
        turn_limit (int): The maximum number of turns in a game.
        num_games (int, optional): The number of games to play for training. Defaults to 100.
    """

    # Create two AI players
    player1 = GameAI(symbol="X", learning_enabled=True)
    player2 = GameAI(symbol="O", learning_enabled=True)

    for _ in range(num_games):
        # Set up and play a game
        game = Game(
            num_pieces=initial_pieces,
            turn_limit=turn_limit,
            player1=player1,
            player2=player2,
        )
        while not game.is_game_over():
            game.play_ai_turn()

        # Determine the outcome of the game
        outcome = game.check_win_condition()

        # Use the outcome to update AI models
        player1.learn_from_game(outcome)
        player2.learn_from_game(outcome)

        print(f"Game {_+1} completed. Outcome: {outcome}")

    print("AI training complete.")


def play_player_vs_ai_game(initial_pieces, turn_limit, app):
    chosen_piece = get_input("Choose piece (1: X, 2: O): ", ["1", "2"])
    player = Player("X") if chosen_piece == "1" else Player("O")
    ai_symbol = "O" if chosen_piece == "1" else "X"
    ai = GameAI(ai_symbol, learning_enabled=False)
    game = Game(initial_pieces, turn_limit, player, ai, random_start=True)
    ai.set_board(game.board)
    gui = GameGUI(game)
    gui.show()
