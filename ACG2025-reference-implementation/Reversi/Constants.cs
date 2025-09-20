namespace ACG2025_reference_implementation.Reversi;

internal static class Constants
{
    public const int BoardSize = 8;
    public const int NumCells = BoardSize * BoardSize;
    public const int NumInitialDiscs = 4;
    public const int NumInitialCells = NumCells - NumInitialDiscs;
    public const int MaxLegalMoves = 33;
}