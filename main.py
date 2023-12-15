import sys
from PyQt5.QtWidgets import QApplication
from functions import *


def main(app):
    print("Starting main function")
    mode = get_input(
        "Select mode (1: Player vs. Player, 2: AI vs. AI, 3: Player vs. AI, 4: Optimize and Train AI): ",
        ["1", "2", "3", "4"],
    )
    initial_pieces = int(
        get_input(
            "Enter the number of pieces every player starts: ",
            [str(i) for i in range(1, 11)],
        )
    )
    turn_limit = int(get_input("Enter the turn limit: ", [str(i) for i in range(1, 101)]))

    if mode == "1":
        play_player_vs_player_game(initial_pieces, turn_limit, app)
    elif mode == "2":
        num_games = int(
            get_input(
                "Enter the number of games for training: ",
                [str(i) for i in range(1, 5001)],
            )
        )
        train_ai_vs_ai(initial_pieces, turn_limit, num_games)  # AI training without GUI
    elif mode == "3":
        play_player_vs_ai_game(initial_pieces, turn_limit, app)
    elif mode == "4":
        num_games_optimize = get_input_int(
            "Enter the number of games for optimization: ", 1, 99999999
        )
        num_trials = get_input_int("Enter the number of trials for optimization: ", 1, 99999999)
        num_games_train = get_input_int("Enter the number of games for training: ", 1, 99999999)
        replay_batch_size = get_input_int(
            "Enter the replay batch size for optimization: ", 1, 99999999
        )
        ai_player1, ai_player2 = optimize_and_train_ai(
            initial_pieces,
            turn_limit,
            num_games_train,
            num_games_optimize,
            num_trials,
            replay_batch_size,
        )

        ai_player1.save_model("ai_player1", initial_pieces)
        ai_player1.save_hyperparameters("ai_player1", initial_pieces)
        ai_player2.save_model("ai_player2", initial_pieces)
        ai_player2.save_hyperparameters("ai_player2", initial_pieces)

    print("Exiting main function")


if __name__ == "__main__":
    print("Starting application")
    app = QApplication([])
    apply_dark_theme(app)
    main(app)
    app.quit()
    print("Application is about to exit")
