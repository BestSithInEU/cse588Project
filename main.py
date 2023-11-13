from game_classes import (
    Game,
    GameGUI,
)
import tkinter as tk


def main():
    initial_pieces = int(input("Enter the number of pieces every player starts: "))
    turn_limit = int(input("Enter the turn limit: "))
    root = tk.Tk()
    game = Game(initial_pieces, turn_limit)
    gui = GameGUI(root, game)
    root.mainloop()


if __name__ == "__main__":
    main()
