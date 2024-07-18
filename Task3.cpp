/*TIC-TAC-TOE AI

Implement an AI agent that plays the classic game of Tic-Tac-Toe
against a human player. You can use algorithms like Minimax with
or without Alpha-Beta Pruning to make the AI player unbeatable.
This project will help you understand game theory and basic search
algorithms.
*/

#include <iostream>
#include <vector>
#include <limits>

const char HUMAN = 'X';
const char AI = 'O';
const char EMPTY = ' ';

// Function to print the board
void printBoard(const std::vector<std::vector<char>>& board) {
    for (const auto& row : board) {
        for (char cell : row) {
            std::cout << cell << " ";
        }
        std::cout << std::endl;
    }
}

// Function to check if the board is full
bool isBoardFull(const std::vector<std::vector<char>>& board) {
    for (const auto& row : board) {
        for (char cell : row) {
            if (cell == EMPTY) {
                return false;
            }
        }
    }
    return true;
}

// Function to evaluate the board
int evaluate(const std::vector<std::vector<char>>& board) {
    // Check rows, columns, and diagonals for a win
    for (int i = 0; i < 3; ++i) {
        if (board[i][0] == board[i][1] && board[i][1] == board[i][2]) {
            if (board[i][0] == AI) return +10;
            else if (board[i][0] == HUMAN) return -10;
        }
        if (board[0][i] == board[1][i] && board[1][i] == board[2][i]) {
            if (board[0][i] == AI) return +10;
            else if (board[0][i] == HUMAN) return -10;
        }
    }
    if (board[0][0] == board[1][1] && board[1][1] == board[2][2]) {
        if (board[0][0] == AI) return +10;
        else if (board[0][0] == HUMAN) return -10;
    }
    if (board[0][2] == board[1][1] && board[1][1] == board[2][0]) {
        if (board[0][2] == AI) return +10;
        else if (board[0][2] == HUMAN) return -10;
    }
    return 0; // No winner
}

// Minimax algorithm function
int minimax(std::vector<std::vector<char>>& board, int depth, bool isMax) {
    int score = evaluate(board);
    if (score == 10 || score == -10 || isBoardFull(board)) {
        return score;
    }
    int bestScore = isMax ? std::numeric_limits<int>::min() : std::numeric_limits<int>::max();
    char player = isMax ? AI : HUMAN;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (board[i][j] == EMPTY) {
                board[i][j] = player;
                int currentScore = minimax(board, depth + 1, !isMax);
                board[i][j] = EMPTY;
                if (isMax) {
                    bestScore = std::max(bestScore, currentScore);
                } else {
                    bestScore = std::min(bestScore, currentScore);
                }
            }
        }
    }
    return bestScore;
}

// Function to find the best move for AI
std::pair<int, int> findBestMove(std::vector<std::vector<char>>& board) {
    int bestScore = std::numeric_limits<int>::min();
    std::pair<int, int> bestMove = {-1, -1};

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (board[i][j] == EMPTY) {
                board[i][j] = AI;
                int moveScore = minimax(board, 0, false);
                board[i][j] = EMPTY;
                if (moveScore > bestScore) {
                    bestScore = moveScore;
                    bestMove = {i, j};
                }
            }
        }
    }
    return bestMove;
}

// Main function to play the game
int main() {
    std::vector<std::vector<char>> board(3, std::vector<char>(3, EMPTY));

    bool isHumanTurn = true;

    while (true) {
        printBoard(board);
        if (evaluate(board) == 10) {
            std::cout << "AI wins!" << std::endl;
            break;
        } else if (evaluate(board) == -10) {
            std::cout << "Human wins!" << std::endl;
            break;
        } else if (isBoardFull(board)) {
            std::cout << "It's a draw!" << std::endl;
            break;
        }

        if (isHumanTurn) {
            int row, col;
            std::cout << "Enter your move (row and column): ";
            std::cin >> row >> col;
            if (board[row][col] == EMPTY) {
                board[row][col] = HUMAN;
                isHumanTurn = false;
            } else {
                std::cout << "Invalid move. Try again." << std::endl;
            }
        } else {
            std::pair<int, int> aiMove = findBestMove(board);
            board[aiMove.first][aiMove.second] = AI;
            isHumanTurn = true;
        }
    }

    printBoard(board);

    return 0;
}
