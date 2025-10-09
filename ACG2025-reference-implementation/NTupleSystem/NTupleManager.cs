namespace ACG2025_reference_implementation.NTupleSystem;

using System;
using System.Linq;

using Reversi;

/// <summary>
/// A structure that manages multiple n-tuples and provides various transformation tables necessary for feature calculation.
/// Holds definition information for each n-tuple and provides data structures and methods for efficient feature extraction 
/// and transformation processing from board positions.
/// </summary>
internal readonly struct NTupleManager
{
    /// <summary>
    /// Gets all the n-tuples being managed.
    /// </summary>
    public readonly ReadOnlySpan<NTuple> NTuples => _ntuples;

    /// <summary>
    /// Gets the number of n-tuples being managed.
    /// </summary>
    public readonly int NumNTuples => _ntuples.Length;

    /// <summary>
    /// Gets the total number of all possible features that can appear in each n-tuple.
    /// </summary>
    public readonly ReadOnlySpan<int> NumFeatures => _numFeatures;

    /// <summary>
    /// Gets the power table for ternary calculations.
    /// The value at index i represents 3^i.
    /// </summary>
    public readonly ReadOnlySpan<int> PowTable => _powTable;

    readonly NTuple[] _ntuples;
    readonly int[] _powTable;
    readonly int[] _numFeatures;
    readonly int[][] _toOpponentFeature;
    readonly int[][] _toMirroredFeature;

    /// <summary>
    /// Initializes NTupleManager with the specified n-tuple array.
    /// Feature transformation tables (opponent perspective transformation, mirror reflection transformation) are also constructed simultaneously.
    /// </summary>
    /// <param name="ntuples">Array of n-tuples to be managed</param>
    public NTupleManager(Span<NTuple> ntuples)
    {
        _ntuples = ntuples.ToArray();

        var powTable = _powTable = new int[_ntuples.Max(x => x.Size) + 1];
        InitPowTable();

        _numFeatures = _ntuples.Select(x => powTable[x.Size]).ToArray();

        _toOpponentFeature = new int[_ntuples.Length][];
        InitOpponentFeatureTable();

        _toMirroredFeature = new int[_ntuples.Length][];
        InitMirroredFeatureTable();
    }

    /// <summary>
    /// Default constructor. Initializes an empty NTupleManager.
    /// </summary>
    public NTupleManager()
    {
        _ntuples = [];
        _powTable = [];
        _numFeatures = [];
        _toOpponentFeature = [];
        _toMirroredFeature = [];
    }

    /// <summary>
    /// Gets the opponent perspective feature transformation table for the specified n-tuple ID.
    /// </summary>
    /// <param name="nTupleID">The ID of the n-tuple</param>
    /// <returns>Opponent perspective feature transformation table</returns>
    public readonly ReadOnlySpan<int> GetOpponentFeatureTable(int nTupleID) => _toOpponentFeature[nTupleID];

    /// <summary>
    /// Gets the mirror reflection feature transformation table for the specified n-tuple ID.
    /// </summary>
    /// <param name="nTupleID">The ID of the n-tuple</param>
    /// <returns>Mirror reflection feature transformation table</returns>
    public readonly ReadOnlySpan<int> GetMirroredFeatureTable(int nTupleID) => _toMirroredFeature[nTupleID];

    internal readonly int[] GetRawOpponentFeatureTable(int nTupleID) => _toOpponentFeature[nTupleID];
    internal readonly int[] GetRawMirroredFeatureTable(int nTupleID) => _toMirroredFeature[nTupleID];

    void InitPowTable()
    {
        _powTable[0] = 1;
        for (var i = 1; i < _powTable.Length; i++)
            _powTable[i] = _powTable[i - 1] * NTuple.NumCellStates;
    }

    void InitOpponentFeatureTable()
    {
        for (var nTupleID = 0; nTupleID < _toOpponentFeature.Length; nTupleID++)
        {
            ref NTuple nTuple = ref _ntuples[nTupleID];
            var table = _toOpponentFeature[nTupleID] = new int[_numFeatures[nTupleID]];
            for (var feature = 0; feature < table.Length; feature++)
            {
                int oppFeature = 0;
                for (var i = 0; i < nTuple.Size; i++)
                {
                    var state = feature / _powTable[i] % NTuple.NumCellStates;
                    if (state == NTuple.EmptyCell)
                        oppFeature += state * _powTable[i];
                    else
                        oppFeature += (int)Utils.ToOpponentColor((DiscColor)state) * _powTable[i];
                }
                table[feature] = oppFeature;
            }
        }
    }

    void InitMirroredFeatureTable()
    {
        for (var nTupleID = 0; nTupleID < _toMirroredFeature.Length; nTupleID++)
        {
            ref NTuple nTuple = ref _ntuples[nTupleID];
            var shuffleTable = nTuple.MirrorTable;

            if (shuffleTable.Length == 0)
            {
                _toMirroredFeature[nTupleID] = [.. from f in Enumerable.Range(0, _numFeatures[nTupleID]) select f];
                continue;
            }

            var table = _toMirroredFeature[nTupleID] = new int[_numFeatures[nTupleID]];
            for (var feature = 0; feature < table.Length; feature++)
            {
                int mirroredFeature = 0;
                for (var i = 0; i < nTuple.Size; i++)
                {
                    var state = feature / _powTable[nTuple.Size - shuffleTable[i] - 1] % NTuple.NumCellStates;
                    mirroredFeature += state * _powTable[nTuple.Size - i - 1];
                }
                table[feature] = mirroredFeature;
            }
        }
    }
}