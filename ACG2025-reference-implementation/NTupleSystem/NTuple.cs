namespace ACG2025_reference_implementation.NTupleSystem;

using System;
using System.Text;
using System.Linq;
using System.Collections.Generic;

using ACG2025_reference_implementation.Reversi;

/// <summary>
/// A structure representing a single n-tuple.
/// Holds the coordinate sequence that represents the n-tuple, including all symmetric forms.
/// </summary>
internal readonly struct NTuple
{
    /// <summary>
    /// Number of possible cell states in the game. Three states: Black, White, and Empty.
    /// </summary>
    public const int NumCellStates = 3; // Three states: (Black: 0, White: 1, Empty: 2)
    /// <summary>
    /// Constant representing a black piece on the board.
    /// </summary>
    public const int BlackCell = 0;
    /// <summary>
    /// Constant representing a white piece on the board.
    /// </summary>
    public const int WhiteCell = 1;
    /// <summary>
    /// Constant representing an empty cell on the board.
    /// </summary>
    public const int EmptyCell = 2;

    /// <summary>
    /// Mirrored order of coordinates that compose the n-tuple. 
    /// The mirroring axis depends on the shape of the n-tuple.
    /// If there is no mirroring axis, MirrorTable is an empty array.
    /// </summary>
    public ReadOnlySpan<int> MirrorTable => _mirrorTable;

    readonly BoardCoordinate[][] _coordinates;
    readonly int[] _mirrorTable;

    /// <summary>
    /// Gets the number of coordinates in the n-tuple.
    /// </summary>
    public int Size => _coordinates[0].Length;
    
    /// <summary>
    /// Gets the number of symmetric expansions (rotations and reflections) for this n-tuple.
    /// </summary>
    public int NumSymmetricExpansions => _coordinates.Length;

    /// <summary>
    /// Initializes a new n-tuple with the specified size using random walk generation.
    /// </summary>
    /// <param name="size">The number of coordinates in the n-tuple</param>
    public NTuple(int size)
    {
        _coordinates = ExpandTuple(InitTupleByRandomWalk(size));
        var coords = _coordinates[0];
        _mirrorTable = [.. from coord in MirrorTuple(coords) select Array.IndexOf(coords, coord)];
    }

    /// <summary>
    /// Initializes a new n-tuple with the specified coordinate array.
    /// </summary>
    /// <param name="coords">Array of board coordinates that compose the n-tuple</param>
    public NTuple(BoardCoordinate[] coords)
    {
        _coordinates = ExpandTuple(coords);
        _mirrorTable = [.. from coord in MirrorTuple(coords) select Array.IndexOf(coords, coord)];
    }

    /// <summary>
    /// Copy constructor. Creates a new n-tuple by copying data from an existing n-tuple.
    /// </summary>
    /// <param name="nTuple">The n-tuple to copy from</param>
    public NTuple(NTuple nTuple)
    {
        _coordinates = new BoardCoordinate[nTuple._coordinates.Length][];

        for (var i = 0; i < _coordinates.Length; i++)
        {
            var srcTuple = nTuple._coordinates[i];
            var destTuple = _coordinates[i] = new BoardCoordinate[srcTuple.Length];
            Buffer.BlockCopy(srcTuple, 0, destTuple, 0, sizeof(BoardCoordinate) * destTuple.Length);
        }

        _mirrorTable = new int[nTuple._mirrorTable.Length];
        Buffer.BlockCopy(nTuple._mirrorTable, 0, _mirrorTable, 0, sizeof(int) * _mirrorTable.Length);
    }

    public static bool operator ==(NTuple lhs, NTuple rhs) => lhs._coordinates.SequenceEqual(rhs._coordinates);
    public static bool operator !=(NTuple lhs, NTuple rhs) => !(lhs == rhs);

    /// <summary>
    /// Gets the coordinates for a specific symmetric expansion of the n-tuple.
    /// </summary>
    /// <param name="idx">Index of the symmetric expansion</param>
    /// <returns>Read-only span of coordinates for the specified symmetric expansion</returns>
    public ReadOnlySpan<BoardCoordinate> GetCoordinates(int idx) => _coordinates[idx];

    /// <summary>
    /// Serializes the n-tuple to a byte array.
    /// </summary>
    /// <returns>Byte array containing the serialized n-tuple data</returns>
    public byte[] ToBytes()
    {
        var size = BitConverter.GetBytes(Size);
        var buffer = new byte[sizeof(int) + Size];
        Buffer.BlockCopy(size, 0, buffer, 0, size.Length);
        for (var i = sizeof(int); i < buffer.Length; i++)
            buffer[i] = (byte)_coordinates[0][i - sizeof(int)];
        return buffer;
    }

    public override bool Equals(object? obj) => obj is NTuple ntuple && this == ntuple;
    public override int GetHashCode() => throw new NotImplementedException();

    /// <summary>
    /// Returns a string representation of the n-tuple visualized on a board grid.
    /// </summary>
    /// <returns>String representation showing the n-tuple coordinates on a board</returns>
    public override readonly string ToString()
    {
        var sb = new StringBuilder();
        sb.Append("  ");
        for (var i = 0; i < Constants.BoardSize; i++)
            sb.Append((char)('A' + i)).Append(' ');

        var tuple = _coordinates[0];
        for (var y = 0; y < Constants.BoardSize; y++)
        {
            sb.Append('\n').Append(y + 1).Append(' ');
            for (var x = 0; x < Constants.BoardSize; x++)
            {
                var idx = Array.IndexOf(tuple, Utils.Coordinate2DTo1D(x, y));
                if (idx != -1)
                    sb.Append(idx).Append(' ');
                else
                    sb.Append("- ");
            }
        }

        return sb.ToString();
    }

    static BoardCoordinate[] InitTupleByRandomWalk(int size)
    {
        var tuple = new List<BoardCoordinate> { (BoardCoordinate)Random.Shared.Next(Constants.NumCells) };
        var adjCoords = Utils.GetAdjacent8Cells(tuple[0]).ToArray().ToList();

        while (tuple.Count < size)
        {
            tuple.Add(adjCoords[Random.Shared.Next(adjCoords.Count)]);
            foreach (var adjCoord in Utils.GetAdjacent8Cells(tuple[^1]))
                adjCoords.Add(adjCoord);
            adjCoords.RemoveAll(tuple.Contains);
        }

        return [.. tuple.Order()];
    }

    /// <summary>
    /// Expands the specified n-tuple (coordinate array) to all symmetric forms by rotation and horizontal reflection,
    /// returning a unique n-tuple array without duplicates.
    /// Specifically, rotates the original tuple 4 times (90 degrees each), and also rotates the horizontally reflected version,
    /// listing all symmetric forms while excluding those that are identical on the board.
    /// </summary>
    /// <param name="ntuple">Base n-tuple (coordinate array)</param>
    /// <returns>n-tuple array containing all symmetric forms by rotation and reflection</returns>
    static BoardCoordinate[][] ExpandTuple(BoardCoordinate[] ntuple)
    {
        var tuples = new List<BoardCoordinate[]>();
        var rotated = new BoardCoordinate[ntuple.Length];
        Buffer.BlockCopy(ntuple, 0, rotated, 0, sizeof(BoardCoordinate) * rotated.Length);

        rotate(rotated);

        for (var j = 0; j < rotated.Length; j++)
            rotated[j] = Utils.HorizontalMirrorCoord[(int)rotated[j]];

        rotate(rotated);

        void rotate(BoardCoordinate[] rotated)
        {
            for (var i = 0; i < 4; i++)
            {
                var ordered = rotated.Order();
                if (!tuples.Any(x => x.Order().SequenceEqual(ordered)))
                {
                    var newTuple = new BoardCoordinate[ntuple.Length];
                    Buffer.BlockCopy(rotated, 0, newTuple, 0, sizeof(BoardCoordinate) * rotated.Length);
                    tuples.Add(newTuple);
                }

                for (var j = 0; j < rotated.Length; j++)
                    rotated[j] = Utils.RotateCoordClockwise[(int)rotated[j]];
            }
        }

        return [.. tuples];
    }

    static BoardCoordinate[] MirrorTuple(BoardCoordinate[] ntuple)
    {
        var mirrored = new BoardCoordinate[ntuple.Length];

        if (mirror(Utils.HorizontalMirrorCoord))
            return mirrored;

        if (mirror(Utils.VerticalMirrorCoord))
            return mirrored;

        if (mirror(Utils.DiagMirrorCoordA1H8))
            return mirrored;

        if (mirror(Utils.DiagMirrorCoordA8H1))
            return mirrored;

        return [];

        bool mirror(ReadOnlySpan<BoardCoordinate> table)
        {
            for (var i = 0; i < mirrored.Length; i++)
                mirrored[i] = table[(int)ntuple[i]];

            if (mirrored.SequenceEqual(ntuple))
                return false;

            return mirrored.Order().SequenceEqual(ntuple.Order());
        }
    }
}