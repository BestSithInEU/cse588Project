import tkinter as tk
from tkinter import messagebox
from .Game import Game


class GameGUI:
    def __init__(self, master, game):
        self.master = master
        self.game = game
        self.selected_piece = None
        self.master.title("AI Board Game")
        self.create_board()

        self.turn_label = tk.Label(self.master, text="Player 1's Turn")
        self.turn_label.grid(row=8, column=0, columnspan=7)

        self.update_board()

    def create_board(self):
        self.buttons = [
            [
                tk.Button(
                    self.master,
                    text=" ",
                    width=4,
                    height=2,
                    command=lambda row=row, col=col: self.on_button_click(row, col),
                )
                for col in range(7)
            ]
            for row in range(7)
        ]
        for row in range(7):
            for col in range(7):
                self.buttons[row][col].grid(row=row, column=col)

    def on_button_click(self, row, col):
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
            if result == "GameOver":
                self.handle_game_over()
            elif result:
                self.update_board()
            self.selected_piece = None  # Reset the selected piece

    def is_valid_selection(self, row, col):
        piece = self.game.board.board[row][col]
        return piece == self.game.current_player.symbol

    def update_board(self):
        print("Updating board...")
        # Loop through the board and update button labels and colors
        for i in range(7):
            for j in range(7):
                button = self.buttons[i][j]
                symbol = self.game.board.board[i][j]
                button["text"] = symbol
                if symbol == "X":
                    button["fg"] = "red"  # Change color for X
                elif symbol == "O":
                    button["fg"] = "blue"  # Change color for O
                else:
                    button["fg"] = "black"  # Default color

        # Update the turn label
        current_turn = (
            "Player 1" if self.game.current_player == self.game.player1 else "Player 2"
        )
        self.turn_label.config(text=f"{current_turn}'s Turn")
        print("Board Updated")

    def handle_game_over(self):
        # Determine the winner or if it's a draw
        winner = self.game.is_game_over()
        if winner:
            winner_message = f"Game Over. Winner: {winner}"
        else:
            winner_message = "Game Over. It's a draw."

        # Display a message box with the game outcome
        messagebox.showinfo("Game Over", winner_message)

        # Ask if players want to restart or exit
        if messagebox.askyesno("Restart", "Would you like to restart the game?"):
            self.reset_game()
        else:
            self.exit_game()

    def reset_game(self):
        # Call reset_game to reset the game state
        self.game.reset_game()

        # Reset the board in the GUI
        for i in range(7):
            for j in range(7):
                self.buttons[i][j]["text"] = " "
                self.buttons[i][j]["fg"] = "black"
                self.buttons[i][j].config(state=tk.NORMAL)

        # Update the GUI to reflect the new game state
        self.update_board()
        self.turn_label.config(text="Player 1's Turn")

    def exit_game(self):
        # Logic to close the application
        self.master.destroy()
