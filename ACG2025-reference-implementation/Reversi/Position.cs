namespace ACG2025_reference_implementation.Reversi;

using System;
using System.Text;
using System.Numerics;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

using ACG2025_reference_implementation.Utils;

/// <summary>
/// Struct representing a position (state) in Reversi.
/// </summary>
internal struct Position
{
    public DiscColor SideToMove
    {
        readonly get => _sideToMove;

        set
        {
            if (_sideToMove != value)
                Pass();
        }
    }

    public DiscColor OpponentColor { get; private set; }

    public readonly int PlayerDiscCount => _bitboard.PlayerDiscCount;
    public readonly int OpponentDiscCount => _bitboard.OpponentDiscCount;
    public readonly int BlackDiscCount => _sideToMove == DiscColor.Black ? PlayerDiscCount : OpponentDiscCount;
    public readonly int WhiteDiscCount => _sideToMove == DiscColor.White ? PlayerDiscCount : OpponentDiscCount;
    public readonly int DiscCount => _bitboard.DiscCount;
    public readonly int DiscDiff => PlayerDiscCount - OpponentDiscCount;
    public int EmptyCellCount { readonly get; private set; }
    public readonly bool CanPass => _bitboard.CalcPlayerMobility() == 0UL && _bitboard.CalcOpponentMobility() != 0UL;
    public readonly bool IsGameOver => _bitboard.CalcPlayerMobility() == 0UL && _bitboard.CalcOpponentMobility() == 0UL;

    Bitboard _bitboard;
    DiscColor _sideToMove;

    public Position()
    {
        _bitboard = new Bitboard(1UL << (int)BoardCoordinate.E4 | 1UL << (int)BoardCoordinate.D5,
                                 1UL << (int)BoardCoordinate.D4 | 1UL << (int)BoardCoordinate.E5);
        _sideToMove = DiscColor.Black;
        OpponentColor = DiscColor.White;
        EmptyCellCount = _bitboard.EmptyCellsCount;
    }

    public Position(Bitboard bitboard, DiscColor sideToMove) => Init(bitboard, sideToMove);

    public readonly Bitboard GetBitboard() => _bitboard;
    public void SetBitboard(Bitboard bitboard) { _bitboard = bitboard; EmptyCellCount = _bitboard.EmptyCellsCount; }
    public readonly bool Has(ref Bitboard bitboard) => _bitboard == bitboard;

    public void Init(Bitboard bitboard, DiscColor sideToMove)
    {
        _bitboard = bitboard;
        _sideToMove = sideToMove;
        OpponentColor = Utils.ToOpponentColor(sideToMove);
        EmptyCellCount = _bitboard.EmptyCellsCount;
    }

    public static bool operator ==(Position left, Position right)
        => left._bitboard == right._bitboard && left._sideToMove == right._sideToMove;

    public static bool operator !=(Position left, Position right)
        => !(left == right);

    /// <summary>
    /// This method is only for suppressing a warning (do not use directly).
    /// </summary>
    /// <param name="obj"></param>
    /// <returns></returns>
    public override readonly bool Equals(object? obj)
        => (obj is Position pos) && (pos == this);

    /// <summary>
    /// This method is only for suppressing a warning (do not use directly).
    /// </summary>
    /// <returns></returns>
    public override readonly int GetHashCode() => (int)ComputeHashCode();

    public void MirrorHorizontal() => _bitboard.MirrorHorizontal();

    public void Rotate90Clockwise() => _bitboard.Rotate90Clockwise();

    public readonly Player GetSquareOwnerAt(BoardCoordinate coord) => _bitboard.GetSquareOwnerAt(coord);

    public readonly DiscColor GetSquareColorAt(BoardCoordinate coord)
    {
        var owner = _bitboard.GetSquareOwnerAt(coord);
        if (owner == Player.Null)
            return DiscColor.Null;
        return owner == Player.First ? _sideToMove : OpponentColor;
    }

    public readonly bool IsLegalMoveAt(BoardCoordinate coord)
        => coord == BoardCoordinate.Pass ? CanPass : (_bitboard.CalcPlayerMobility() & (1UL << (int)coord)) != 0UL;

    public readonly int GetScore(DiscColor color) => (color == _sideToMove) ? DiscDiff : -DiscDiff;

    public void Pass()
    {
        (_sideToMove, OpponentColor) = (OpponentColor, _sideToMove);
        _bitboard.Swap();
    }

    public void PutPlayerDiscAt(BoardCoordinate coord)
    {
        _bitboard.PutPlayerDiscAt(coord);
        EmptyCellCount = _bitboard.EmptyCellsCount;
    }

    public void PutOpponentDiscAt(BoardCoordinate coord)
    {
        _bitboard.PutOpponentDiscAt(coord);
        EmptyCellCount = _bitboard.EmptyCellsCount;
    }

    public void PutDisc(DiscColor color, BoardCoordinate coord)
    {
        if (color == DiscColor.Null)
            return;

        if (_sideToMove == color)
            PutPlayerDiscAt(coord);
        else
            PutOpponentDiscAt(coord);
    }

    public void RemoveDiscAt(BoardCoordinate coord)
    {
        _bitboard.RemoveDiscAt(coord);
        EmptyCellCount = _bitboard.EmptyCellsCount;
    }

    public void RemoveAllDiscs()
    {
        for (var coord = BoardCoordinate.A1; coord <= BoardCoordinate.H8; coord++)
            RemoveDiscAt(coord);
    }

    public void Update(Move move) => Update(ref move);

    /// <summary>
    /// Update the position by a specified move without checking legality.
    /// Pass moves are not supported. Use the Position.Pass method for pass moves.
    /// </summary>
    /// <param name="move"></param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Update(ref Move move)
    {
        (_sideToMove, OpponentColor) = (OpponentColor, _sideToMove);
        _bitboard.Update(move.Coord, move.Flip);
        EmptyCellCount--;
    }

    /// <summary>
    /// Update the position by making a move at the specified coordinate.
    /// BoardCoordinate.Pass means a pass move.
    /// This method may fail if an illegal coordinate is specified.
    /// </summary>
    /// <param name="coord"></param>
    /// <returns>True if the move at the specified coordinate is legal, otherwise false.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool Update(BoardCoordinate coord)
    {
        if (!IsLegalMoveAt(coord))
            return false;

        if (coord == BoardCoordinate.Pass)
        {
            Pass();
            return true;
        }

        ulong flip = _bitboard.CalcFlip(coord);
        var move = new Move(coord, flip);
        Update(ref move);
        return true;
    }

    /// <summary>
    /// Undo the specified move without checking its legality.
    /// If an illegal move is specified, the position may become inconsistent.
    /// </summary>
    /// <param name="move"></param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Undo(ref Move move)
    {
        (_sideToMove, OpponentColor) = (OpponentColor, _sideToMove);
        _bitboard.Undo(move.Coord, move.Flip);
        EmptyCellCount++;
    }

    /// <summary>
    /// Store all current legal moves into the given buffer (moves).
    /// </summary>
    /// <param name="moves">Buffer to store legal moves</param>
    /// <returns>The number of legal moves</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly int GetNextMoves(ref Span<Move> moves)
    {
        ulong mobility = _bitboard.CalcPlayerMobility();
        var moveCount = 0;
        for (var coord = BitManipulations.FindFirstSet(mobility); mobility != 0; coord = BitManipulations.FindNextSet(ref mobility))
            moves[moveCount++].Coord = (BoardCoordinate)coord;
        return moveCount;
    }

    public readonly IEnumerable<BoardCoordinate> EnumerateNextMoves()
    {
        foreach (var coord in BitManipulations.EnumerateSets(_bitboard.CalcPlayerMobility()))
            yield return (BoardCoordinate)coord;
    }

    public readonly int GetNumNextMoves() => BitOperations.PopCount(_bitboard.CalcPlayerMobility());

    public readonly Move CalcFlip(BoardCoordinate coord) => new(coord, _bitboard.CalcFlip(coord));

    public readonly void CalcFlip(ref Move move) => move.Flip = _bitboard.CalcFlip(move.Coord);

    public readonly ulong ComputeHashCode() => _bitboard.ComputeHashCode();

    public override readonly string ToString()
    {
        var sb = new StringBuilder();
        sb.Append("  ");
        for (var i = 0; i < Constants.BoardSize; i++)
            sb.Append((char)('A' + i)).Append(' ');

        (ulong p, ulong o) = (_bitboard.Player, _bitboard.Opponent);
        var mask = 1UL;
        for (var y = 0; y < Constants.BoardSize; y++)
        {
            sb.Append('\n').Append(y + 1).Append(' ');
            for (var x = 0; x < Constants.BoardSize; x++)
            {
                if ((p & mask) != 0UL)
                    sb.Append((_sideToMove == DiscColor.Black) ? '*' : 'O').Append(' ');
                else if ((o & mask) != 0UL)
                    sb.Append((OpponentColor == DiscColor.Black) ? '*' : 'O').Append(' ');
                else
                    sb.Append("- ");
                mask <<= 1;
            }
        }

        return sb.ToString();
    }
}