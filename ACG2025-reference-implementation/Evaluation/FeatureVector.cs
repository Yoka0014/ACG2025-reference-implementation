global using FeatureType = System.UInt16;

namespace ACG2025_reference_implementation.Evaluation;

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

using ACG2025_reference_implementation.NTupleSystem;
using ACG2025_reference_implementation.Reversi;
using ACG2025_reference_implementation.Utils;

using static ACG2025_reference_implementation.Reversi.Constants;

/// <summary>
/// A structure that stores feature values for all symmetric expansions of a single n-tuple.
/// Each element corresponds to a specific symmetric form (rotation/reflection) of the n-tuple,
/// containing the encoded feature value calculated from the board position for that symmetric expansion.
/// </summary>
internal unsafe struct Feature
{
    /// <summary>
    /// Maximum number of symmetric expansions that can be stored for a single n-tuple.
    /// </summary>
    const int MaxLen = 8;

    /// <summary>
    /// Fixed array to store feature values for each symmetric expansion of the n-tuple.
    /// </summary>
    fixed FeatureType _values[MaxLen];

    /// <summary>
    /// Gets or sets the feature value for the specified symmetric expansion index.
    /// </summary>
    /// <param name="idx">The zero-based index of the symmetric expansion</param>
    /// <returns>The encoded feature value for the specified symmetric expansion</returns>
    /// <exception cref="IndexOutOfRangeException">Thrown when idx is less than 0 or greater than or equal to Length (in DEBUG builds)</exception>
    public FeatureType this[int idx]
    {
        get
        {
#if DEBUG
            if (idx < 0 || idx >= Length)
                throw new IndexOutOfRangeException();
#endif

            return _values[idx];
        }

        set
        {
#if DEBUG
            if (idx < 0 || idx >= Length)
                throw new IndexOutOfRangeException();
#endif

            _values[idx] = value;
        }
    }

    /// <summary>
    /// Gets the number of symmetric expansions being stored for this n-tuple.
    /// </summary>
    public int Length { get; private set; }

    /// <summary>
    /// Initializes a new instance of the Feature structure with the specified number of symmetric expansions.
    /// </summary>
    /// <param name="length">The number of symmetric expansions for the n-tuple</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when length is less than 0 or greater than MaxLen</exception>
    public Feature(int length)
    {
        if (length < 0 || length > MaxLen)
            throw new ArgumentOutOfRangeException(nameof(length));
        Length = length;
    }

    /// <summary>
    /// Copies all feature values from this instance to the specified destination Feature.
    /// This includes both the feature values for all symmetric expansions and the length.
    /// </summary>
    /// <param name="dest">The destination Feature to copy values to</param>
    public void CopyTo(ref Feature dest)
    {
        for (var i = 0; i < Length; i++)
            dest._values[i] = _values[i];
        dest.Length = Length;
    }
}

/// <summary>
/// A class that manages feature vectors for position evaluation.
/// Handles the calculation and incremental update of features extracted from board positions
/// using multiple n-tuples. Provides efficient methods for updating features when moves are made or undone.
/// </summary>
internal class FeatureVector
{
    /// <summary>
    /// Gets an empty FeatureVector instance for default initialization.
    /// </summary>
    public static FeatureVector Empty => _Empty;
    static readonly FeatureVector _Empty = new();

    /// <summary>
    /// Gets the current player whose turn it is to move.
    /// </summary>
    public DiscColor SideToMove { get; private set; }
    
    /// <summary>
    /// Gets the number of empty cells remaining on the board.
    /// </summary>
    public int EmptyCellCount { get; private set; }
    
    /// <summary>
    /// Gets the number of n-tuples managed by this feature vector.
    /// </summary>
    public int NumNTuples => NTupleManager.NumNTuples;
    
    /// <summary>
    /// Gets the n-tuple manager that defines the n-tuples used for feature extraction.
    /// </summary>
    public NTupleManager NTupleManager { get; }

    /// <summary>
    /// Gets the array of features, one for each n-tuple.
    /// </summary>
    public Feature[] Features { get; private set; }  
    readonly FeatureDiff[] _featureDiffTable = new FeatureDiff[NumCells];

    delegate void Updator(ref Move move);
    Updator _playerUpdator;
    Updator _opponentUpdator;
    Updator _playerRestorer;
    Updator _opponentRestorer;

    /// <summary>
    /// Initializes a new instance of FeatureVector with the specified n-tuple manager.
    /// </summary>
    /// <param name="nTupleManager">The n-tuple manager that defines the n-tuples for feature extraction</param>
    public FeatureVector(NTupleManager nTupleManager)
    {
        NTupleManager = nTupleManager;
        Features = new Feature[NTupleManager.NumNTuples];
        var nTuples = NTupleManager.NTuples;
        for (var nTupleID = 0; nTupleID < Features.Length; nTupleID++)
            Features[nTupleID] = new Feature(nTuples[nTupleID].NumSymmetricExpansions);

        (_playerUpdator, _opponentUpdator) = (Update<Black>, Update<White>);
        (_playerRestorer, _opponentRestorer) = (Undo<White>, Undo<Black>);
        
        InitFeatureDiffTable();
    }

    /// <summary>
    /// Private constructor for creating an empty FeatureVector.
    /// </summary>
    FeatureVector()
    {
        NTupleManager = new NTupleManager();
        Features = [];
        _featureDiffTable = [];
        (_playerUpdator, _opponentUpdator) = (Update<Black>, Update<White>);
        (_playerRestorer, _opponentRestorer) = (Undo<White>, Undo<Black>);
    }

    /// <summary>
    /// Copy constructor. Creates a new FeatureVector by copying data from an existing one.
    /// </summary>
    /// <param name="featureVec">The FeatureVector to copy from</param>
    public FeatureVector(FeatureVector featureVec)
    {
        SideToMove = featureVec.SideToMove;
        EmptyCellCount = featureVec.EmptyCellCount;
        NTupleManager = featureVec.NTupleManager;
        Features = new Feature[featureVec.NumNTuples];
        for(var i = 0; i < Features.Length; i++)
            featureVec.Features[i].CopyTo(ref Features[i]);

        if (SideToMove == DiscColor.Black)
        {
            (_playerUpdator, _opponentUpdator) = (Update<Black>, Update<White>);
            (_playerRestorer, _opponentRestorer) = (Undo<White>, Undo<Black>);
        }
        else
        {
            (_playerUpdator, _opponentUpdator) = (Update<White>, Update<Black>);
            (_playerRestorer, _opponentRestorer) = (Undo<Black>, Undo<White>);
        }

        _featureDiffTable = featureVec._featureDiffTable;
    }

    void InitFeatureDiffTable()
    {
        var diffs = new List<(int nTupleID, int idx, FeatureType diff)>();
        var tuples = NTupleManager.NTuples;
        for (var coord = BoardCoordinate.A1; coord <= BoardCoordinate.H8; coord++)
        {
            diffs.Clear();
            for (var nTupleID = 0; nTupleID < tuples.Length; nTupleID++)
            {
                for (var idx = 0; idx < tuples[nTupleID].NumSymmetricExpansions; idx++)
                {
                    var coords = tuples[nTupleID].GetCoordinates(idx);
                    var coordIdx = Array.IndexOf(coords.ToArray(), coord);
                    if (coordIdx != -1)
                        diffs.Add((nTupleID, idx, (FeatureType)NTupleManager.PowTable[coords.Length - coordIdx - 1]));
                }
            }
            _featureDiffTable[(int)coord].Values = [.. diffs];
        }
    }

    /// <summary>
    /// Gets a reference to the feature for the specified n-tuple ID.
    /// </summary>
    /// <param name="nTupleID">The ID of the n-tuple</param>
    /// <returns>Reference to the feature for the specified n-tuple</returns>
    public ref Feature GetFeature(int nTupleID) => ref Features[nTupleID];

    /// <summary>
    /// Initializes the feature vector with the specified board position and legal moves.
    /// </summary>
    /// <param name="bitboard">The current board position</param>
    /// <param name="sideToMove">The player whose turn it is to move</param>
    public void Init(Bitboard bitboard, DiscColor sideToMove)
    {
        var pos = new Position(bitboard, sideToMove);
        Init(ref pos);
    }

    /// <summary>
    /// Initializes the feature vector with the specified position and legal moves.
    /// </summary>
    /// <param name="pos">Reference to the current position</param>
    public void Init(ref Position pos)
    {
        SideToMove = pos.SideToMove;
        EmptyCellCount = pos.EmptyCellCount;

        if (SideToMove == DiscColor.Black)
        {
            (_playerUpdator, _opponentUpdator) = (Update<Black>, Update<White>);
            (_playerRestorer, _opponentRestorer) = (Undo<White>, Undo<Black>);
        }
        else
        {
            (_playerUpdator, _opponentUpdator) = (Update<White>, Update<Black>);
            (_playerRestorer, _opponentRestorer) = (Undo<Black>, Undo<White>);
        }

        for (var nTupleID = 0; nTupleID < Features.Length; nTupleID++)
        {
            ReadOnlySpan<NTuple> nTuples = NTupleManager.NTuples;
            ref Feature f = ref Features[nTupleID];
            for (var i = 0; i < nTuples[nTupleID].NumSymmetricExpansions; i++)
            {
                f[i] = 0;
                var coordinates = nTuples[nTupleID].GetCoordinates(i);
                foreach (BoardCoordinate coord in coordinates)
                    f[i] = (FeatureType)(f[i] * NTuple.NumCellStates + (int)pos.GetSquareColorAt(coord));
            }
        }
    }

    public unsafe void CopyTo(FeatureVector dest)
    {
        dest.SideToMove = SideToMove;
        dest.EmptyCellCount = EmptyCellCount;
        if (dest.SideToMove == DiscColor.Black)
        {
            (dest._playerUpdator, dest._opponentUpdator) = (dest.Update<Black>, dest.Update<White>);
            (dest._playerRestorer, dest._opponentRestorer) = (dest.Undo<White>, dest.Undo<Black>);
        }
        else
        {
            (dest._playerUpdator, dest._opponentUpdator) = (dest.Update<White>, dest.Update<Black>);
            (dest._playerRestorer, dest._opponentRestorer) = (dest.Undo<Black>, dest.Undo<White>);
        }

        fixed (Feature* features = Features)
        fixed (Feature* destFeatures = dest.Features)
        {
            for (var i = 0; i < NTupleManager.NumNTuples; i++)
                features[i].CopyTo(ref destFeatures[i]);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Update(ref Move move)
    {
        _playerUpdator.Invoke(ref move);
        (_playerUpdator, _opponentUpdator) = (_opponentUpdator, _playerUpdator);
        (_playerRestorer, _opponentRestorer) = (_opponentRestorer, _playerRestorer);
        EmptyCellCount--;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Undo(ref Move move)
    {
        _playerRestorer.Invoke(ref move);
        (_playerUpdator, _opponentUpdator) = (_opponentUpdator, _playerUpdator);
        (_playerRestorer, _opponentRestorer) = (_opponentRestorer, _playerRestorer);
        EmptyCellCount++;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Pass()
    {
        SideToMove = Utils.ToOpponentColor(SideToMove);
        (_playerUpdator, _opponentUpdator) = (_opponentUpdator, _playerUpdator);
        (_playerRestorer, _opponentRestorer) = (_opponentRestorer, _playerRestorer);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    unsafe void Update<SideToMove>(ref Move move) where SideToMove : struct, IDiscColor
    {
        var placer = typeof(SideToMove) == typeof(Black) ? NTuple.BlackCell - NTuple.EmptyCell : NTuple.WhiteCell - NTuple.EmptyCell;
        var flipper = typeof(SideToMove) == typeof(Black) ? NTuple.BlackCell - NTuple.WhiteCell : NTuple.WhiteCell - NTuple.BlackCell;

        fixed(Feature* features = Features)
        fixed (FeatureDiff* featureDiffTable = _featureDiffTable)
        {
            foreach (var (nTupleID, idx, diff) in featureDiffTable[(int)move.Coord].Values)
                features[nTupleID][idx] += (FeatureType)(placer * diff);

            ulong flip = move.Flip;
            for (int coord = BitManipulations.FindFirstSet(flip); flip != 0; coord = BitManipulations.FindNextSet(ref flip))
            {
                foreach (var (nTupleID, idx, diff) in featureDiffTable[coord].Values)
                    features[nTupleID][idx] += (FeatureType)(flipper * diff);
            }
        }

        this.SideToMove = typeof(SideToMove) == typeof(Black) ? DiscColor.White : DiscColor.Black;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    unsafe void Undo<SideToMove>(ref Move move) where SideToMove : struct, IDiscColor
    {
        var remover = typeof(SideToMove) == typeof(Black) ? NTuple.EmptyCell - NTuple.BlackCell : NTuple.EmptyCell - NTuple.WhiteCell;
        var flipper = typeof(SideToMove) == typeof(Black) ? NTuple.WhiteCell - NTuple.BlackCell : NTuple.BlackCell - NTuple.WhiteCell;

        fixed (Feature* features = Features)
        fixed (FeatureDiff* featureDiffTable = _featureDiffTable)
        {
            foreach (var (nTupleID, idx, diff) in featureDiffTable[(int)move.Coord].Values)
                features[nTupleID][idx] += (FeatureType)(remover * diff);

            ulong flipped = move.Flip;
            for (int coord = BitManipulations.FindFirstSet(flipped); flipped != 0; coord = BitManipulations.FindNextSet(ref flipped))
            {
                foreach (var (nTupleID, idx, diff) in featureDiffTable[coord].Values)
                    features[nTupleID][idx] += (FeatureType)(flipper * diff);
            }
        }

        this.SideToMove = typeof(SideToMove) == typeof(Black) ? DiscColor.Black : DiscColor.White;
    }

    /// <summary>
    /// A structure that stores feature difference information for a specific board coordinate.
    /// Contains tuples of n-tuple ID, symmetric expansion index, and the difference value
    /// to be applied when the coordinate state changes.
    /// </summary>
    struct FeatureDiff
    {
        /// <summary>
        /// Gets or sets the array of tuples containing n-tuple ID, symmetric expansion index, 
        /// and difference value for feature updates.
        /// </summary>
        public (int NTupleID, int Idx, FeatureType Diff)[] Values { get; set; }
    }
}