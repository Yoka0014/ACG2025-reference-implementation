namespace ACG2025_reference_implementation.Reversi;

internal struct Move(BoardCoordinate coord, ulong flip)
{
    public static ref readonly Move Pass => ref _Pass;
    public static ref readonly Move Null => ref _Null;

    static readonly Move _Pass = new (BoardCoordinate.Pass);
    static readonly Move _Null = new (BoardCoordinate.Null);

    public BoardCoordinate Coord { get; set; } = coord;
    public ulong Flip { get; set; } = flip;

    public Move() : this(BoardCoordinate.Null, 0UL) { }
    public Move(BoardCoordinate coord) : this(coord, 0UL) { }
}
