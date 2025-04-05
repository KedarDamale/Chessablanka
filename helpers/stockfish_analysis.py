import chess
import chess.engine
import os
from typing import Dict, List, Optional, Union

def analyze_position(fen: str, stockfish_path: str, depth: int = 22) -> Dict[str, Union[str, List[str]]]:
    """
    Analyze a chess position using Stockfish engine.
    
    Args:
        fen (str): FEN string representing the chess position
        stockfish_path (str): Path to Stockfish executable
        depth (int): Analysis depth (default: 15)
    
    Returns:
        Dict containing evaluation, best move and principal variation
    """
    engine = None
    try:
        # Verify Stockfish exists
        if not os.path.exists(stockfish_path):
            raise FileNotFoundError(f"Stockfish executable not found at: {stockfish_path}")

        # Initialize engine with specific configuration
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"Threads": 1, "Hash": 128})
        board = chess.Board(fen)

        # Verify position is valid
        if not board.is_valid():
            raise ValueError(f"Invalid chess position: {fen}")

        # Run analysis without MultiPV option
        info = engine.analyse(
            board,
            chess.engine.Limit(depth=depth, time=2.0)
        )

        # Extract evaluation
        score = info["score"] if "score" in info else None
        if score:
            if score.is_mate():
                mate_in = score.relative.mate()
                eval_str = f"Mate in {abs(mate_in)}" + (" for White" if mate_in > 0 else " for Black")
            else:
                cp_score = score.relative.score()
                if cp_score is not None:
                    eval_str = f"{cp_score/100:+.2f}"
                else:
                    eval_str = "0.00"
        else:
            eval_str = "0.00"

        # Extract best move and line
        best_move = "No best move found"
        best_line = []

        if "pv" in info:
            moves = info["pv"]
            board_copy = board.copy()
            
            if moves:
                try:
                    # Get best move
                    best_move = board.san(moves[0])
                    
                    # Get principal variation (up to 5 moves)
                    for move in moves[:5]:
                        san_move = board_copy.san(move)
                        best_line.append(san_move)
                        board_copy.push(move)
                except (chess.IllegalMoveError, chess.InvalidMoveError):
                    pass

        return {
            "evaluation": eval_str,
            "best_move": best_move,
            "best_line": best_line
        }

    except chess.engine.EngineTerminatedError:
        raise Exception("Stockfish engine terminated unexpectedly")
    except chess.engine.EngineError as e:
        raise Exception(f"Stockfish engine error: {e}")
    except Exception as e:
        raise Exception(f"Analysis error: {str(e)}")
    finally:
        if engine:
            engine.quit()