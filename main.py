import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from helpers.get_board_layout import *
from helpers.map_grid import *
from helpers.detection import *
from helpers.piece_square_mapping import *
from helpers.map_to_chessboard import *
from helpers.stockfish_analysis import analyze_position
import sys
import io
import chess

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    """
    Main function to run the script.
    """
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Select 4 points on an image and perform perspective transform')
    parser.add_argument('--image', '-i', type=str, default="images/1.jpg", 
                        help='Path to the input image')
    parser.add_argument('--output', '-o', type=str, default="process_output", 
                        help='Directory to save the output images')
    parser.add_argument('--size', '-s', type=int, default=800,
                        help='Size of the output square image (width=height)')
    parser.add_argument('--highlight', '-hl', type=str, default=None,
                        help='Chess square to highlight (e.g., E4)')
    
    args = parser.parse_args()
    
    # Ensure the input file exists
    if not os.path.isfile(args.image):
        print(f"Error: Input file '{args.image}' not found.")
        return
    
    # Ensure the output directory exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)  # Create the output directory if it doesn't exist
    
    try:
        # Load the original image for preview
        original_image = cv2.imread(args.image)
        
        # Get user points and crop
        output_size = (args.size, args.size)
        points, cropped = get_user_points_and_crop(args.image, output_size)
        
        if points is None or cropped is None:
            print("Operation cancelled.")
            return
        
        # Save the cropped image
        cropped_image_path = os.path.join(args.output, "cropped_board.jpg")
        cv2.imwrite(cropped_image_path, cropped)
        print(f"Cropped image saved as '{cropped_image_path}'")
        
        # Draw the chess grid on the cropped board
        square_coords = get_square_coordinates(output_size)
        board_with_grid = draw_chess_grid(cropped, output_size, square_coords)
        
        # Highlight a specific square if requested
        if args.highlight:
            board_with_grid = highlight_square(board_with_grid, args.highlight, output_size)
        
        # Save the image with the chess grid
        mapped_grid_image_path = os.path.join(args.output, "mapped_grid_board.jpg")
        cv2.imwrite(mapped_grid_image_path, board_with_grid)
        print(f"Mapped grid board saved as '{mapped_grid_image_path}'")
        
        # Save the square coordinates to a file
        square_coords_file = os.path.join(args.output, "square_coordinates.txt")
        save_square_coordinates(square_coords, square_coords_file)
        
        # Run the Roboflow model to detect chess pieces
        detect_chess_pieces(
            image_path=cropped_image_path,
            output_dir=args.output,
            api_key="Hc5MzZYyVof2Gw4hrAGc",
            model_id="chesspiecedetection-ptwhz",
            version=3
        )

        # Map pieces to squares
        piece_to_square_mapping = map_pieces_to_squares(
            piece_coordinates_path=os.path.join(args.output, "cropped_board_coordinates.txt"),
            square_coordinates_path=os.path.join(args.output, "square_coordinates.txt")
        )

        # Save the mapping to a file
        mapping_file = os.path.join(args.output, "piece_square_mapping.txt")
        with open(mapping_file, 'w') as f:
            for square, piece in piece_to_square_mapping.items():
                f.write(f"{square}: {piece}\n")
        
        # Print the mapping to the console with better formatting
        print("\n" + "="*50)
        print("📍 DETECTED PIECES AND THEIR POSITIONS")
        print("="*50)

        # Separate pieces by color
        white_pieces = []
        black_pieces = []
        for square, piece in sorted(piece_to_square_mapping.items()):
            piece_name = piece.split('-')[1].capitalize()
            if piece.startswith("white"):
                white_pieces.append(f"{piece_name:6} at {square}")
            else:
                black_pieces.append(f"{piece_name:6} at {square}")

        # Find the maximum length of the two lists
        max_len = max(len(white_pieces), len(black_pieces))

        # Print header
        print("\n{:<20} {:<20}".format("White Pieces ⚪", "Black Pieces ⚫"))
        print("="*40)

        # Print pieces side by side
        for i in range(max_len):
            white = white_pieces[i] if i < len(white_pieces) else " "*15
            black = black_pieces[i] if i < len(black_pieces) else " "*15
            print(f"{white:<20} {black:<20}")
        
        # Render the chessboard in CLI with better formatting
        print("\n" + "="*50)
        print("🎯 CURRENT BOARD POSITION")
        print("="*50)
        print(render_chessboard_cli(piece_to_square_mapping))
        
        print(f"\n📝 Piece-to-square mapping saved to '{mapping_file}'")

        # Generate and display FEN with better formatting
        fen = generate_fen_from_mapping(piece_to_square_mapping)
        print("\n" + "="*50)
        print("🔍 POSITION DETAILS")
        print("="*50)
        print(f"FEN: {fen}")

        # Validate the FEN string and show turn
        try:
            board = chess.Board(fen)
            if not board.is_valid():
                print("❌ The generated FEN is invalid.")
                return
            
            # Print whose move it is with emoji
            player = "White ⚪" if board.turn else "Black ⚫"
            print(f"\n👉 It is {player}'s turn to move")
            
        except ValueError as e:
            print(f"❌ Error validating FEN: {e}")
            return

        # Run Stockfish analysis with improved output
        stockfish_path = os.path.join(os.getcwd(), "stockfish", "stockfish-windows-x86-64.exe")
        try:
            print("\n" + "="*50)
            print("🤖 CHESSABLANKA ANALYSIS")
            print("="*50)
            print("Analyzing position...")
            analysis = analyze_position(fen, stockfish_path=stockfish_path, depth=22)

            # Print analysis results with better formatting
            if analysis['evaluation'] != "0.00":
                eval_color = "⚪" if float(analysis['evaluation'].replace('+', '')) > 0 else "⚫"
                print(f"\n📊 Evaluation: {eval_color} {analysis['evaluation']}")
            else:
                print("\n📊 Position is equal (0.00)")
                
            if analysis['best_move'] != "No best move found":
                print(f"\n💡 Best move: {analysis['best_move']}")
                if analysis['best_line']:
                    print(f"📈 Best line: {' → '.join(analysis['best_line'])}")
            else:
                print("\n❌ No legal moves found in this position")
                
        except FileNotFoundError:
            print(f"\n❌ Error: Stockfish not found at {stockfish_path}")
            print("⚠️  Please ensure Stockfish is installed in the correct location")
        except Exception as e:
            print(f"\n❌ Stockfish Analysis Error: {str(e)}")
            print("⚠️  Continuing without engine analysis...")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()