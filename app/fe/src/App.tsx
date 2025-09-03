import React, { useState } from "react";

interface GameState {
  sessionId: null | string;
  board: number[][];
  gameOver: boolean;
  winner: number | null;
}

interface MoveResponse {
  move: number[];
  winner: number | null;
  session_id: string;
}

const HexGame: React.FC = () => {
  const [gameState, setGameState] = useState<GameState>({
    sessionId: null,
    board: Array(5)
      .fill(null)
      .map(() => Array(5).fill(0)),
    gameOver: false,
    winner: null,
  });

  const [selectedModel, setSelectedModel] = useState<string>("ppo");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");

  const resetGame = async () => {
    setGameState({
      sessionId: null,
      board: Array(5)
        .fill(null)
        .map(() => Array(5).fill(0)),
      gameOver: false,
      winner: null,
    });
  };

  const makeMove = async (row: number, col: number) => {
    if (gameState.board[row][col] !== 0 || gameState.gameOver || isLoading) {
      return;
    }

    // setIsLoading(true);
    setGameState((prev) => {
      const board: any = [...prev.board];
      board[row][col] = 1;

      return {
        ...prev,
        board,
      };
    });
    setError("");

    try {
      const response = await fetch("http://0.0.0.0:8000/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: gameState.sessionId,
          move: [row, col],
          // row: row,
          // col: col,
          model: selectedModel,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to make move");
      }
      const data: MoveResponse = await response.json();

      if (data.winner == 1) {
        setGameState((prev) => ({
          ...prev,
          winner: 1,
          gameOver: true,
        }));

        return;
      }

      setGameState((prev) => {
        const board = [...prev.board];
        board[data.move[0]][data.move[1]] = 2;

        return {
          sessionId: data.session_id,
          board,
          gameOver: data.winner !== null,
          winner: data.winner,
        };
      });
    } catch (err) {
      setError("Failed to make move. Check your connection.");
    } finally {
      setIsLoading(false);
    }
  };

  const getCellClass = (row: number, col: number) => {
    const cell = gameState.board[row][col];
    let baseClass =
      "w-12 h-12 border-2 border-gray-400 cursor-pointer transition-all duration-200 flex items-center justify-center text-white font-bold";

    if (cell === 1) {
      baseClass += " bg-blue-500 hover:bg-blue-600";
    } else if (cell === 2) {
      baseClass += " bg-red-500 hover:bg-red-600";
    } else {
      baseClass += " bg-gray-200 hover:bg-gray-300";
    }

    if (gameState.gameOver || isLoading) {
      baseClass += " cursor-not-allowed opacity-25";
    }
    return baseClass;
  };

  const getHexStyle = (row: number) => ({
    marginLeft: `${row * 1.5}rem`,
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            5x5 Hex Game AI
          </h1>
          <p className="text-lg text-gray-600">
            Play against trained AI models! Blue connects top-bottom, Red
            connects left-right.
          </p>

          {/* GitHub + Experiment note */}
          <div className="mt-4 flex items-center justify-center gap-3">
            <span className="inline-flex items-center rounded-full bg-indigo-100 px-3 py-1 text-sm font-medium text-indigo-700">
              Experiment: AlphaZero, PPO
            </span>
            <a
              href="https://github.com/SameerNaing/hex-rl"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 rounded-md bg-gray-900 px-4 py-2 text-white hover:bg-gray-800 transition-colors"
              aria-label="View source on GitHub"
              title="View source on GitHub"
            >
              {/* GitHub icon (SVG) */}
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="currentColor"
                className="h-5 w-5"
              >
                <path d="M12 .5A11.5 11.5 0 0 0 .5 12c0 5.08 3.29 9.38 7.86 10.9.58.1.8-.25.8-.57v-2.06c-3.2.7-3.88-1.37-3.88-1.37-.53-1.35-1.3-1.72-1.3-1.72-1.06-.72.08-.7.08-.7 1.18.08 1.8 1.22 1.8 1.22 1.04 1.78 2.73 1.27 3.4.98.1-.76.4-1.27.73-1.56-2.55-.29-5.23-1.28-5.23-5.71 0-1.26.45-2.28 1.2-3.08-.12-.3-.52-1.53.11-3.19 0 0 .98-.31 3.2 1.18a11 11 0 0 1 5.82 0c2.22-1.49 3.2-1.18 3.2-1.18.63 1.66.23 2.89.11 3.19.75.8 1.2 1.82 1.2 3.08 0 4.44-2.68 5.41-5.24 5.7.41.35.77 1.03.77 2.08v3.08c0 .32.21.68.81.56A11.5 11.5 0 0 0 23.5 12 11.5 11.5 0 0 0 12 .5z" />
              </svg>
              <span>View Source</span>
            </a>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="flex flex-wrap items-center justify-between mb-6 gap-4">
            <div className="flex items-center gap-4">
              <label className="text-lg font-medium text-gray-700">
                AI Model:
              </label>
              <select
                value={selectedModel}
                onChange={(e) => {
                  setSelectedModel(e.target.value);
                  resetGame();
                }}
                className="px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                disabled={isLoading}
              >
                <option value="ppo">PPO</option>
                <option value="ppo_alphazero">PPO + AlphaZero</option>
              </select>
            </div>

            <button
              onClick={resetGame}
              disabled={isLoading}
              className="px-6 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              New Game
            </button>
          </div>

          {error && (
            <div className="mb-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded-md">
              {error}
            </div>
          )}

          <div className="flex items-center justify-center mb-6">
            <div className="relative">
              {/* Top border for Blue */}
              <div className="absolute -top-4 left-0 right-0 h-2 bg-blue-500 rounded"></div>
              {/* Bottom border for Blue */}
              <div className="absolute -bottom-4 left-0 right-0 h-2 bg-blue-500 rounded"></div>
              {/* Left border for Red */}
              <div className="absolute -left-4 top-0 bottom-0 w-2 bg-red-500 rounded"></div>
              {/* Right border for Red */}
              <div className="absolute -right-4 top-0 bottom-0 w-2 bg-red-500 rounded"></div>

              <div className="hex-board p-4">
                {gameState.board.map((row, rowIndex) => (
                  <div
                    key={rowIndex}
                    className="flex gap-1 mb-1"
                    style={getHexStyle(rowIndex)}
                  >
                    {row.map((cell, colIndex) => (
                      <button
                        key={`${rowIndex}-${colIndex}`}
                        className={getCellClass(rowIndex, colIndex)}
                        onClick={() => makeMove(rowIndex, colIndex)}
                        disabled={gameState.gameOver || isLoading || cell !== 0}
                      >
                        {[1, 2].includes(cell) && "●"}
                      </button>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="text-center">
            {isLoading && (
              <div className="text-lg text-blue-600 font-medium">
                AI is thinking...
              </div>
            )}

            {!gameState.gameOver && !isLoading && (
              <div className="text-lg font-medium">
                Current Turn:
                <span className="text-blue-600 ml-2">You (Blue)</span>
              </div>
            )}

            {gameState.gameOver && (
              <div className="text-xl font-bold">
                {gameState.winner === 1 ? (
                  <span className="text-blue-600">You Win! 🎉</span>
                ) : gameState.winner === 2 ? (
                  <span className="text-red-600">AI Wins! 🤖</span>
                ) : (
                  <span className="text-gray-600">Draw!</span>
                )}
              </div>
            )}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">How to Play</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-lg font-semibold text-blue-600 mb-2">
                Your Goal (Blue)
              </h3>
              <p className="text-gray-700">
                Connect the top and bottom edges of the board with your blue
                pieces.
              </p>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-red-600 mb-2">
                AI Goal (Red)
              </h3>
              <p className="text-gray-700">
                The AI tries to connect the left and right edges with red
                pieces.
              </p>
            </div>
          </div>
          <div className="mt-4 p-4 bg-gray-50 rounded-md">
            <p className="text-sm text-gray-600">
              <strong>Models:</strong> PPO (Proximal Policy Optimization),
              PPO+AlphaZero (Hybrid approach) - each with different playing
              styles and strengths!
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HexGame;
