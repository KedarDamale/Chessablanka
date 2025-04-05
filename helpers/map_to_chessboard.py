def render_chessboard_cli(piece_to_square_mapping):
    """
    Render a simple text-based chessboard using ASCII characters.

    Args:
        piece_to_square_mapping (dict): Mapping of squares to piece names (like 'white-pawn').
    """
    piece_symbols = {
        "white-pawn": "P",
        "black-pawn": "p",
        "white-rook": "R",
        "black-rook": "r",
        "white-knight": "N",
        "black-knight": "n",
        "white-bishop": "B",
        "black-bishop": "b",
        "white-queen": "Q",
        "black-queen": "q",
        "white-king": "K",
        "black-king": "k",
        "empty": "."
    }

    # Create an 8x8 empty board
    board = [["empty" for _ in range(8)] for _ in range(8)]

    for square, piece in piece_to_square_mapping.items():
        if len(square) == 2:
            col = ord(square[0].lower()) - ord('a')
            row = 8 - int(square[1])
            board[row][col] = piece

    # Print the board
    print("White pices are uppercase, black pieces are lowercase.")
    print("  a b c d e f g h")
    for i, row in enumerate(board):
        print(8 - i, end=" ")
        for piece in row:
            print(piece_symbols.get(piece, "."), end=" ")
        print(8 - i)
    print("  a b c d e f g h")


def print_chess_position(fen=None, piece_mapping=None):
    """
    Print a chess position from FEN or piece mapping.
    """
    if fen and not piece_mapping:
        piece_mapping = {}
        symbol_to_piece = {
            'p': 'black-pawn', 'P': 'white-pawn',
            'r': 'black-rook', 'R': 'white-rook',
            'n': 'black-knight', 'N': 'white-knight',
            'b': 'black-bishop', 'B': 'white-bishop',
            'q': 'black-queen', 'Q': 'white-queen',
            'k': 'black-king', 'K': 'white-king'
        }

        board_fen = fen.split()[0]
        row = 0
        col = 0

        for char in board_fen:
            if char == '/':
                row += 1
                col = 0
            elif char.isdigit():
                col += int(char)
            elif char in symbol_to_piece:
                square = chr(col + ord('a')) + str(8 - row)
                piece_mapping[square] = symbol_to_piece[char]
                col += 1

    render_chessboard_cli(piece_mapping or {})


def generate_fen_from_mapping(piece_to_square_mapping):
    """
    Generate a FEN string from the piece-to-square mapping.
    
    Args:
        piece_to_square_mapping (dict): Mapping of squares to pieces.
        
    Returns:
        str: FEN string representing the position.
    """
    piece_symbols = {
        "white-pawn": "P", "black-pawn": "p",
        "white-rook": "R", "black-rook": "r",
        "white-knight": "N", "black-knight": "n",
        "white-bishop": "B", "black-bishop": "b",
        "white-queen": "Q", "black-queen": "q",
        "white-king": "K", "black-king": "k",
    }

    # Create an 8x8 board
    board = [["" for _ in range(8)] for _ in range(8)]

    # Map pieces to board
    for square, piece in piece_to_square_mapping.items():
        col = ord(square[0].lower()) - ord('a')
        row = 8 - int(square[1])
        board[row][col] = piece_symbols.get(piece, "")

    # Generate FEN rows
    fen_rows = []
    for row in board:
        empty_count = 0
        fen_row = ""
        for cell in row:
            if cell == "":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += cell
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)

    # Get player to move from user
    while True:
        player_to_move = input("\nWho moves next? (w for White, b for Black): ").lower()
        if player_to_move in ['w', 'b']:
            break
        print("Invalid input. Please enter 'w' for White or 'b' for Black.")

    # Combine rows into FEN with the correct player to move
    fen = "/".join(fen_rows) + f" {player_to_move} - - 0 1"
    return fen
