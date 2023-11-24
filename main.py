from PyQt5.QtWidgets import (
    QApplication,
)

# from PyQt5.QtGui import (
#     QPalette,
#     QColor,
# )
# from PyQt5.QtCore import Qt
# from game_classes import (
#     Game,
#     GameGUI,
#     GameAI,
#     Player,
# )
from functions import *


def main(app):
    mode = get_input(
        "Select mode (1: Player vs. Player, 2: AI vs. AI, 3: Player vs. AI): ",
        ["1", "2", "3"],
    )
    initial_pieces = int(
        get_input(
            "Enter the number of pieces every player starts: ",
            [str(i) for i in range(1, 11)],
        )
    )
    turn_limit = int(
        get_input("Enter the turn limit: ", [str(i) for i in range(1, 101)])
    )

    if mode == "1":
        play_player_vs_player_game(initial_pieces, turn_limit, app)
    elif mode == "2":
        num_games = int(
            get_input(
                "Enter the number of games for training: ",
                [str(i) for i in range(1, 5001)],
            )
        )
        train_ai_vs_ai(initial_pieces, turn_limit, num_games)
    elif mode == "3":
        play_player_vs_ai_game(initial_pieces, turn_limit, app)

    app.exec_()


if __name__ == "__main__":
    app = QApplication([])
    apply_dark_theme(app)
    main(app)
