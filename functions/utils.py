from PyQt5.QtWidgets import (
    QApplication,
    QInputDialog,
)
from PyQt5.QtGui import (
    QPalette,
    QColor,
)
from PyQt5.QtCore import Qt
from classes import (
    Game,
    GameGUI,
    GameAI,
    Player,
)
from classes.Game import GameConfig
import numpy as np
from typing import (
    Tuple,
    Dict,
    List,
    Optional,
)


def get_input(prompt: str, valid_options: List[str]) -> str:
    """
    Displays a dialog box to prompt the user for input and returns the selected choice.

    Parameters:
    - prompt (str): The message displayed in the dialog box to prompt the user for input.
    - valid_options (List[str]): The list of valid options that the user can choose from.

    Returns:
    - str: The selected choice from the dialog box.

    Raises:
    - ValueError: If valid_options contains items that are not strings.
    """
    options_text = valid_options
    choice, _ = QInputDialog.getItem(None, "Input", prompt, options_text, editable=False)
    return choice


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


def play_player_vs_player_game(initial_pieces: int, turn_limit: int, app: QApplication) -> None:
    """
    Play a player vs player game.

    Args:
    - initial_pieces (int): The initial pieces for the game.
    - turn_limit (int): The maximum number of turns allowed.
    - app (QApplication): The Qt application.

    Returns:
        None
    """
    # Create a new instance of the Game class with the initial pieces and turn limit
    game = Game(initial_pieces, turn_limit)

    # Create a new instance of the GameGUI class with the game instance
    gui = GameGUI(game)

    # Show the game GUI
    gui.show()

    # Start the Qt application event loop
    app.exec_()


def play_player_vs_ai_game(initial_pieces: int, turn_limit: int, app: QApplication) -> None:
    """
    Play a game between a human player and an AI player.

    Args:
    - initial_pieces (int): The initial state of the game board.
    - turn_limit (int): The maximum number of turns allowed in the game.
    - app (QApplication): The main application object.

    Returns:
        None
    """
    chosen_piece = get_input("Choose piece (1: X, 2: O): ", ["1", "2"])
    player_symbol = "X" if chosen_piece == "1" else "O"
    ai_symbol = "O" if chosen_piece == "1" else "X"

    player = Player(player_symbol)
    ai = GameAI(ai_symbol, learning_enabled=False)
    config = GameConfig(initial_pieces, turn_limit, random_start=True)
    game = Game(config, player, ai)
    ai.set_game(game)

    gui = GameGUI(game)

    # Show the game GUI
    gui.show()

    # Start the Qt application event loop
    app.exec_()


def train_ai_vs_ai(initial_pieces: int, turn_limit: int, num_games: int = 100) -> None:
    """
    Trains two AI players against each other by playing a specified number of games.

    Args:
    - initial_pieces (int): The number of pieces each player starts with.
    - turn_limit (int): The maximum number of turns in the game.
    - num_games (int, optional): Number of games to be played for training. Defaults to 100.
    """
    # Initialize two AI players
    ai_player1 = GameAI("X", learning_enabled=True)
    ai_player2 = GameAI("O", learning_enabled=True)

    # Create a game instance with the two AI players
    config = GameConfig(initial_pieces, turn_limit, random_start=True)
    game = Game(config, ai_player1, ai_player2)

    ai_player1.set_game(game)
    ai_player2.set_game(game)

    # Train the AI players by playing the specified number of games
    game.train_ai(num_games)

    # Optional: Save or further process the trained models
    # ai_player1.save_model('path_to_save_player1_model')
    # ai_player2.save_model('path_to_save_player2_model')


def calculate_individual_performance_metric(
    game_results: List[Tuple[str, str]], ai_symbol: str
) -> float:
    """
    Calculate performance metric based on game outcomes for an individual AI player.

    Args:
    - game_results: List of game results with ('result', 'winner_symbol').
    - ai_symbol: The symbol of the AI player for which the performance is calculated.

    Returns:
    - float: A performance score for the individual AI player.
    """
    win_count = sum(result == "win" and winner == ai_symbol for result, winner in game_results)
    loss_count = sum(result == "lose" and winner != ai_symbol for result, winner in game_results)
    draw_count = sum(result == "draw" for result, winner in game_results)

    # Scoring: win = 1 point, loss = -0.5 points, draw = 0.2 points
    score = win_count * 1.0 - loss_count * 0.5 + draw_count * 0.2
    return score


def calculate_performance_metric(game_results: List[str]) -> float:
    """
    Calculate a performance metric based on game outcomes.

    Args:
    game_results (List[str]): List of game results ('win', 'lose', 'draw', etc.)

    Returns:
    float: A performance score.
    """
    win_count = game_results.count("win")
    loss_count = game_results.count("lose")
    draw_count = game_results.count("draw")

    score = win_count * 1.0 - loss_count * 0.5 + draw_count * 0.2
    return score


def run_game_simulation(game: Game, num_games: int) -> Tuple[float, float]:
    """
    Run a simulation of the game for a given number of games and calculate performance metrics for each player.

    Args:
        game (Game): The game object.
        num_games (int): The number of games to simulate.

    Returns:
        Tuple[float, float]: A tuple containing the performance metric for player 1 and player 2, respectively.
    """
    game_results: List[Tuple[str, str]] = []

    for game_index in range(num_games):
        game.reset_game()
        print(f"\nStarting Game {game_index + 1}")

        while not game.is_game_over():
            print(
                f"Turn {game.current_turn}: Player {game.current_player.symbol} is making a move."
            )
            game.play_ai_turn()

        game_result: str = game.determine_game_result()
        winner_symbol: Optional[str] = game.check_win_condition()
        winner_symbol = winner_symbol if winner_symbol is not None else "No Winner"
        game_results.append((game_result, winner_symbol))
        print(f"Game {game_index + 1} Result: {game_result}, Winner: {winner_symbol}")

    performance_metric_player1: float = calculate_individual_performance_metric(
        game_results, game.player1.symbol
    )
    performance_metric_player2: float = calculate_individual_performance_metric(
        game_results, game.player2.symbol
    )

    print(f"Performance Metric Player 1: {performance_metric_player1}")
    print(f"Performance Metric Player 2: {performance_metric_player2}")

    return performance_metric_player1, performance_metric_player2


def hyperparameter_search(
    num_trials: int,
    ai_player1: GameAI,
    ai_player2: GameAI,
    initial_pieces: int,
    turn_limit: int,
    num_games: int,
) -> Tuple[GameAI, Dict[str, float]]:
    """
    Runs a hyperparameter optimization process with specified ranges.

    Args:
    - num_trials (int): The number of different hyperparameter sets to test.
    - ai_player1 (GameAI): The AI player object.
    - ai_player2 (GameAI): The AI player object.
    - initial_pieces (int): Number of initial pieces for the game.
    - turn_limit (int): The turn limit for the game.
    - num_games (int): Number of games to simulate for each hyperparameter set.

    Returns:
    - Tuple[GameAI, Dict[str, float]]: Tuple containing the best AI player
    - and the best found hyperparameters.
    """
    best_performance_player1 = -float("inf")
    best_performance_player2 = -float("inf")
    best_hyperparams_player1 = {}
    best_hyperparams_player2 = {}

    for _ in range(num_trials):
        hyperparams = {
            "gamma": np.random.uniform(0.5, 1.0),
            "learning_rate": np.random.uniform(0.0001, 0.01),
            "learning_rate_decay": np.random.uniform(0.990, 1.0),
        }

        ai_player1.set_hyperparameters(hyperparams)
        ai_player2.set_hyperparameters(hyperparams)

        config = GameConfig(initial_pieces, turn_limit, random_start=True)

        performance_player1, performance_player2 = run_game_simulation(
            Game(config, ai_player1, ai_player2),
            num_games,
        )

        if performance_player1 > best_performance_player1:
            best_performance_player1 = performance_player1
            best_hyperparams_player1 = hyperparams

        if performance_player2 > best_performance_player2:
            best_performance_player2 = performance_player2
            best_hyperparams_player2 = hyperparams

    if best_performance_player1 > best_performance_player2:
        print("Best AI player: Player 1", "Hyperparameters: ", best_hyperparams_player1)
        return ai_player1, best_hyperparams_player1
    else:
        print("Best AI player: Player 2", "Hyperparameters: ", best_hyperparams_player2)
        return ai_player2, best_hyperparams_player2


def optimize_and_train_ai(
    initial_pieces: int,
    turn_limit: int,
    num_games_train: int,
    num_games_optimize: int,
    num_trials: int,
) -> Tuple[GameAI, GameAI]:
    """
    Optimizes hyperparameters and then trains the AI.

    Args:
    - initial_pieces (int): Initial number of pieces.
    - turn_limit (int): Turn limit for the game.
    - num_games_train (int): Number of games for training.
    - num_games_optimize (int): Number of games for each hyperparameter optimization trial.
    - num_trials (int): Number of trials for hyperparameter optimization.

    Returns:
    - Tuple[GameAI, GameAI]: A tuple containing the two AI players.
    """
    # Initialize AI players
    ai_player1 = GameAI("X", learning_enabled=True)
    ai_player2 = GameAI("O", learning_enabled=True)

    # Create a game instance with the two AI players
    config = GameConfig(initial_pieces, turn_limit, random_start=True)
    game = Game(config, ai_player1, ai_player2)
    ai_player1.set_game(game)
    ai_player2.set_game(game)

    # Hyperparameter optimization
    print("Optimizing hyperparameters...")
    _, optimized_hyperparams = hyperparameter_search(
        num_trials,
        ai_player1,
        ai_player2,
        initial_pieces,
        turn_limit,
        num_games_optimize,
    )
    ai_player1.set_hyperparameters(optimized_hyperparams)
    ai_player2.set_hyperparameters(optimized_hyperparams)
    print(f"Optimized hyperparameters: {optimized_hyperparams}")

    return ai_player1, ai_player2
