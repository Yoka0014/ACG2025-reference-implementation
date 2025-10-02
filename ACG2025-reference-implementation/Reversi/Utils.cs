namespace ACG2025_reference_implementation.Reversi;

using System;
using System.Collections.Generic;

using static Constants;

internal static class Utils
{
    /// <summary>
    /// Table to convert a board cell coordinate to its horizontally mirrored coordinate.
    /// e.g. HorizontalMirrorCoord[(int)BoardCoordinate.A1] := BoardCoordinate.H1
    /// </summary>
    public static ReadOnlySpan<BoardCoordinate> HorizontalMirrorCoord =>
    [
        BoardCoordinate.H1, BoardCoordinate.G1, BoardCoordinate.F1, BoardCoordinate.E1, BoardCoordinate.D1, BoardCoordinate.C1, BoardCoordinate.B1, BoardCoordinate.A1,
        BoardCoordinate.H2, BoardCoordinate.G2, BoardCoordinate.F2, BoardCoordinate.E2, BoardCoordinate.D2, BoardCoordinate.C2, BoardCoordinate.B2, BoardCoordinate.A2,
        BoardCoordinate.H3, BoardCoordinate.G3, BoardCoordinate.F3, BoardCoordinate.E3, BoardCoordinate.D3, BoardCoordinate.C3, BoardCoordinate.B3, BoardCoordinate.A3,
        BoardCoordinate.H4, BoardCoordinate.G4, BoardCoordinate.F4, BoardCoordinate.E4, BoardCoordinate.D4, BoardCoordinate.C4, BoardCoordinate.B4, BoardCoordinate.A4,
        BoardCoordinate.H5, BoardCoordinate.G5, BoardCoordinate.F5, BoardCoordinate.E5, BoardCoordinate.D5, BoardCoordinate.C5, BoardCoordinate.B5, BoardCoordinate.A5,
        BoardCoordinate.H6, BoardCoordinate.G6, BoardCoordinate.F6, BoardCoordinate.E6, BoardCoordinate.D6, BoardCoordinate.C6, BoardCoordinate.B6, BoardCoordinate.A6,
        BoardCoordinate.H7, BoardCoordinate.G7, BoardCoordinate.F7, BoardCoordinate.E7, BoardCoordinate.D7, BoardCoordinate.C7, BoardCoordinate.B7, BoardCoordinate.A7,
        BoardCoordinate.H8, BoardCoordinate.G8, BoardCoordinate.F8, BoardCoordinate.E8, BoardCoordinate.D8, BoardCoordinate.C8, BoardCoordinate.B8, BoardCoordinate.A8
    ];

    /// <summary>
    /// Table to convert a board cell coordinate to its vertically mirrored coordinate.
    /// e.g. VerticalMirrorCoord[(int)BoardCoordinate.A1] := BoardCoordinate.A8
    /// </summary>
    public static ReadOnlySpan<BoardCoordinate> VerticalMirrorCoord =>
    [
        BoardCoordinate.A8, BoardCoordinate.B8, BoardCoordinate.C8, BoardCoordinate.D8, BoardCoordinate.E8, BoardCoordinate.F8, BoardCoordinate.G8, BoardCoordinate.H8,
        BoardCoordinate.A7, BoardCoordinate.B7, BoardCoordinate.C7, BoardCoordinate.D7, BoardCoordinate.E7, BoardCoordinate.F7, BoardCoordinate.G7, BoardCoordinate.H7,
        BoardCoordinate.A6, BoardCoordinate.B6, BoardCoordinate.C6, BoardCoordinate.D6, BoardCoordinate.E6, BoardCoordinate.F6, BoardCoordinate.G6, BoardCoordinate.H6,
        BoardCoordinate.A5, BoardCoordinate.B5, BoardCoordinate.C5, BoardCoordinate.D5, BoardCoordinate.E5, BoardCoordinate.F5, BoardCoordinate.G5, BoardCoordinate.H5,
        BoardCoordinate.A4, BoardCoordinate.B4, BoardCoordinate.C4, BoardCoordinate.D4, BoardCoordinate.E4, BoardCoordinate.F4, BoardCoordinate.G4, BoardCoordinate.H4,
        BoardCoordinate.A3, BoardCoordinate.B3, BoardCoordinate.C3, BoardCoordinate.D3, BoardCoordinate.E3, BoardCoordinate.F3, BoardCoordinate.G3, BoardCoordinate.H3,
        BoardCoordinate.A2, BoardCoordinate.B2, BoardCoordinate.C2, BoardCoordinate.D2, BoardCoordinate.E2, BoardCoordinate.F2, BoardCoordinate.G2, BoardCoordinate.H2,
        BoardCoordinate.A1, BoardCoordinate.B1, BoardCoordinate.C1, BoardCoordinate.D1, BoardCoordinate.E1, BoardCoordinate.F1, BoardCoordinate.G1, BoardCoordinate.H1
    ];

    /// <summary>
    /// Table to convert a board cell coordinate to its diagonal mirror along the A1-H8 line.
    /// e.g. DiagMirrorCoordA1H8[(int)BoardCoordinate.A8] := BoardCoordinate.H1
    /// </summary>
    public static ReadOnlySpan<BoardCoordinate> DiagMirrorCoordA1H8 =>
    [
        BoardCoordinate.A1, BoardCoordinate.A2, BoardCoordinate.A3, BoardCoordinate.A4, BoardCoordinate.A5, BoardCoordinate.A6, BoardCoordinate.A7, BoardCoordinate.A8,
        BoardCoordinate.B1, BoardCoordinate.B2, BoardCoordinate.B3, BoardCoordinate.B4, BoardCoordinate.B5, BoardCoordinate.B6, BoardCoordinate.B7, BoardCoordinate.B8,
        BoardCoordinate.C1, BoardCoordinate.C2, BoardCoordinate.C3, BoardCoordinate.C4, BoardCoordinate.C5, BoardCoordinate.C6, BoardCoordinate.C7, BoardCoordinate.C8,
        BoardCoordinate.D1, BoardCoordinate.D2, BoardCoordinate.D3, BoardCoordinate.D4, BoardCoordinate.D5, BoardCoordinate.D6, BoardCoordinate.D7, BoardCoordinate.D8,
        BoardCoordinate.E1, BoardCoordinate.E2, BoardCoordinate.E3, BoardCoordinate.E4, BoardCoordinate.E5, BoardCoordinate.E6, BoardCoordinate.E7, BoardCoordinate.E8,
        BoardCoordinate.F1, BoardCoordinate.F2, BoardCoordinate.F3, BoardCoordinate.F4, BoardCoordinate.F5, BoardCoordinate.F6, BoardCoordinate.F7, BoardCoordinate.F8,
        BoardCoordinate.G1, BoardCoordinate.G2, BoardCoordinate.G3, BoardCoordinate.G4, BoardCoordinate.G5, BoardCoordinate.G6, BoardCoordinate.G7, BoardCoordinate.G8,
        BoardCoordinate.H1, BoardCoordinate.H2, BoardCoordinate.H3, BoardCoordinate.H4, BoardCoordinate.H5, BoardCoordinate.H6, BoardCoordinate.H7, BoardCoordinate.H8
    ];

    /// <summary>
    /// Table to convert a board cell coordinate to its diagonal mirror along the A8-H1 line.
    /// e.g. DiagMirrorCoordA8H1[(int)BoardCoordinate.A1] := BoardCoordinate.H8
    /// </summary>
    public static ReadOnlySpan<BoardCoordinate> DiagMirrorCoordA8H1 =>
    [
        BoardCoordinate.H8, BoardCoordinate.H7, BoardCoordinate.H6, BoardCoordinate.H5, BoardCoordinate.H4, BoardCoordinate.H3, BoardCoordinate.H2, BoardCoordinate.H1,
        BoardCoordinate.G8, BoardCoordinate.G7, BoardCoordinate.G6, BoardCoordinate.G5, BoardCoordinate.G4, BoardCoordinate.G3, BoardCoordinate.G2, BoardCoordinate.G1,
        BoardCoordinate.F8, BoardCoordinate.F7, BoardCoordinate.F6, BoardCoordinate.F5, BoardCoordinate.F4, BoardCoordinate.F3, BoardCoordinate.F2, BoardCoordinate.F1,
        BoardCoordinate.E8, BoardCoordinate.E7, BoardCoordinate.E6, BoardCoordinate.E5, BoardCoordinate.E4, BoardCoordinate.E3, BoardCoordinate.E2, BoardCoordinate.E1,
        BoardCoordinate.D8, BoardCoordinate.D7, BoardCoordinate.D6, BoardCoordinate.D5, BoardCoordinate.D4, BoardCoordinate.D3, BoardCoordinate.D2, BoardCoordinate.D1,
        BoardCoordinate.C8, BoardCoordinate.C7, BoardCoordinate.C6, BoardCoordinate.C5, BoardCoordinate.C4, BoardCoordinate.C3, BoardCoordinate.C2, BoardCoordinate.C1,
        BoardCoordinate.B8, BoardCoordinate.B7, BoardCoordinate.B6, BoardCoordinate.B5, BoardCoordinate.B4, BoardCoordinate.B3, BoardCoordinate.B2, BoardCoordinate.B1,
        BoardCoordinate.A8, BoardCoordinate.A7, BoardCoordinate.A6, BoardCoordinate.A5, BoardCoordinate.A4, BoardCoordinate.A3, BoardCoordinate.A2, BoardCoordinate.A1
    ];

    /// <summary>
    /// Table to convert a board cell coordinate to its coordinate after a clockwise rotation.
    /// e.g. RotateCoordClockwise[(int)BoardCoordinate.A2] := BoardCoordinate.G1
    /// </summary>
    public static ReadOnlySpan<BoardCoordinate> RotateCoordClockwise =>
    [
        BoardCoordinate.H1, BoardCoordinate.H2, BoardCoordinate.H3, BoardCoordinate.H4, BoardCoordinate.H5, BoardCoordinate.H6, BoardCoordinate.H7, BoardCoordinate.H8,
        BoardCoordinate.G1, BoardCoordinate.G2, BoardCoordinate.G3, BoardCoordinate.G4, BoardCoordinate.G5, BoardCoordinate.G6, BoardCoordinate.G7, BoardCoordinate.G8,
        BoardCoordinate.F1, BoardCoordinate.F2, BoardCoordinate.F3, BoardCoordinate.F4, BoardCoordinate.F5, BoardCoordinate.F6, BoardCoordinate.F7, BoardCoordinate.F8,
        BoardCoordinate.E1, BoardCoordinate.E2, BoardCoordinate.E3, BoardCoordinate.E4, BoardCoordinate.E5, BoardCoordinate.E6, BoardCoordinate.E7, BoardCoordinate.E8,
        BoardCoordinate.D1, BoardCoordinate.D2, BoardCoordinate.D3, BoardCoordinate.D4, BoardCoordinate.D5, BoardCoordinate.D6, BoardCoordinate.D7, BoardCoordinate.D8,
        BoardCoordinate.C1, BoardCoordinate.C2, BoardCoordinate.C3, BoardCoordinate.C4, BoardCoordinate.C5, BoardCoordinate.C6, BoardCoordinate.C7, BoardCoordinate.C8,
        BoardCoordinate.B1, BoardCoordinate.B2, BoardCoordinate.B3, BoardCoordinate.B4, BoardCoordinate.B5, BoardCoordinate.B6, BoardCoordinate.B7, BoardCoordinate.B8,
        BoardCoordinate.A1, BoardCoordinate.A2, BoardCoordinate.A3, BoardCoordinate.A4, BoardCoordinate.A5, BoardCoordinate.A6, BoardCoordinate.A7, BoardCoordinate.A8
    ];

    /// <summary>
    /// Table giving the coordinates of the 4 adjacent (orthogonal: up, down, left, right) cells for a given board cell.
    /// </summary>
    readonly static BoardCoordinate[][] Adjacent4Cells = new BoardCoordinate[NumCells][];

    /// <summary>
    /// Table giving the coordinates of the 8 adjacent (orthogonal and diagonal) cells for a given board cell.
    /// </summary>
    readonly static BoardCoordinate[][] Adjacent8Cells = new BoardCoordinate[NumCells][];

    static Utils()
    {
        Span<(int x, int y)> dirs4 = [(1, 0), (-1, 0), (0, 1), (0, -1)];
        Span<(int x, int y)> dirsDiag = [(1, 1), (-1, 1), (1, -1), (-1, -1)];

        var cells = new List<BoardCoordinate>();
        for (var y = 0; y < BoardSize; y++)
            for (var x = 0; x < BoardSize; x++)
            {
                var coord = Coordinate2DTo1D(x, y);
                cells.Clear();
                foreach (var (dx, dy) in dirs4)
                {
                    var (adjX, adjY) = (x + dx, y + dy);
                    if (adjX >= 0 && adjX < BoardSize && adjY >= 0 && adjY < BoardSize)
                        cells.Add(Coordinate2DTo1D(adjX, adjY));
                }
                Adjacent4Cells[(int)coord] = [.. cells];

                foreach (var (dx, dy) in dirsDiag)
                {
                    var (adjX, adjY) = (x + dx, y + dy);
                    if (adjX >= 0 && adjX < BoardSize && adjY >= 0 && adjY < BoardSize)
                        cells.Add(Coordinate2DTo1D(adjX, adjY));
                }
                Adjacent8Cells[(int)coord] = [.. cells];
            }
    }

    public static ReadOnlySpan<BoardCoordinate> GetAdjacent4Cells(BoardCoordinate coord) => Adjacent4Cells[(int)coord];
    public static ReadOnlySpan<BoardCoordinate> GetAdjacent8Cells(BoardCoordinate coord) => Adjacent8Cells[(int)coord];

    public static (int x, int y) Coordinate1DTo2D(BoardCoordinate coord) => ((int)coord % BoardSize, (int)coord / BoardSize);

    public static BoardCoordinate Coordinate2DTo1D(int x, int y)
    {
        if (x < 0 || y < 0 || x >= BoardSize || y >= BoardSize)
            throw new ArgumentOutOfRangeException($"Coordinate (x, y) was out of range within [(0, 0), ({BoardSize - 1}, {BoardSize - 1}].");

        return (BoardCoordinate)(x + y * BoardSize);
    }

    public static BoardCoordinate ParseCoordinate(string? str)
    {
        if (str is null)
            return BoardCoordinate.Null;

        var lstr = str.Trim().ToLower();

        if (lstr == "pass" || lstr == "pa")
            return BoardCoordinate.Pass;

        if (lstr.Length < 2 || lstr[0] < 'a' || lstr[0] > ('a' + BoardSize - 1) || lstr[1] < '1' || lstr[1] > ('1' + BoardSize - 1))
            return BoardCoordinate.Null;

        return Coordinate2DTo1D(lstr[0] - 'a', lstr[1] - '1');
    }

    public static string? CoordinateToString(BoardCoordinate coord)
    {
        if (coord == BoardCoordinate.Pass)
            return "pass";

        var (x, y) = Coordinate1DTo2D(coord);
        if (BoardCoordinate.A1 <= coord && coord <= BoardCoordinate.H8)
            return $"{(char)('A' + x)}{(char)('1' + y)}";
            
        return null;
    }

    public static DiscColor ToOpponentColor(DiscColor color) => color ^ DiscColor.White;

    public static DiscColor ParseDiscColor(string? str)
    {
        if (str is null)
            return DiscColor.Null;

        var lstr = str.Trim().ToLower();

        if (lstr == "b" || lstr == "black")
            return DiscColor.Black;

        else if (lstr == "w" || lstr == "white")
            return DiscColor.White;

        return DiscColor.Null;
    }

    public static Player ToOpponentPlayer(Player player) => player ^ Player.Second;

    public static GameResult ToOpponentGameResult(GameResult res)
    {
        if (res == GameResult.NotOver)
            return res;
        return (GameResult)(-(int)res);
    }
}