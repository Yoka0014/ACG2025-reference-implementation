namespace ACG2025_reference_implementation.Search.AlphaBetaPruning;

using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using ACG2025_reference_implementation.Utils;
using ACG2025_reference_implementation.Reversi;

/// <summary>
/// Represents an entry in the transposition table, storing position information,
/// evaluation bounds, best move, and metadata for replacement policies.
/// </summary>
internal struct TTEntry
{
    /// <summary>
    /// The bitboard representation of the position.
    /// </summary>
    public Bitboard Position;
    
    /// <summary>
    /// The lower bound of the evaluation (alpha bound).
    /// </summary>
    public short Lower;
    
    /// <summary>
    /// The upper bound of the evaluation (beta bound).
    /// </summary>
    public short Upper;
    
    /// <summary>
    /// The best move found for this position, if any.
    /// </summary>
    public BoardCoordinate Move;
    
    /// <summary>
    /// The generation number when this entry was created, used for aging.
    /// </summary>
    public byte Generation;
    
    /// <summary>
    /// The logarithmic cost (node count) of the search that created this entry.
    /// </summary>
    public byte Cost;
    
    /// <summary>
    /// The search depth at which this entry was created.
    /// </summary>
    public byte Depth;

    /// <summary>
    /// Gets the priority value used for entry replacement decisions.
    /// Higher priority entries are less likely to be replaced.
    /// </summary>
    public readonly int Priority => (Generation << 16) | (Cost << 8) | Depth;
    
    /// <summary>
    /// Gets a value indicating whether this entry contains an exact evaluation (best move).
    /// True when Lower equals Upper, indicating the search found the exact value.
    /// </summary>
    public readonly bool HasBestMove => Lower == Upper;

    /// <summary>
    /// Initializes a new instance of the TTEntry struct.
    /// </summary>
    public TTEntry() { }
}

/// <summary>
/// Represents a cluster of transposition table entries that share the same hash index.
/// Uses a 4-way set-associative design to reduce hash collisions while maintaining cache efficiency.
/// </summary>
[InlineArray(NumEntries)]
internal struct TTCluster
{
    /// <summary>
    /// The number of entries in each cluster (4-way set-associative).
    /// </summary>
    public const int NumEntries = 4;

    TTEntry _entry;

#pragma warning disable CS9181 // Suppress warning for using custom indexer for performance optimization
    /// <summary>
    /// Gets a reference to the entry at the specified index within the cluster.
    /// Uses unsafe code for high-performance access during search operations.
    /// </summary>
    /// <param name="idx">The index of the entry to access</param>
    /// <returns>A reference to the TTEntry at the specified index</returns>
    /// <exception cref="IndexOutOfRangeException">Thrown in debug builds when the index is out of range</exception>
    public unsafe ref TTEntry this[int idx]
    {
        get
        {
#if DEBUG
            if (idx < 0 || idx > NumEntries)
                throw new IndexOutOfRangeException();
#endif

            fixed (TTCluster* self = &this)
            {
                var entries = (TTEntry*)self;
                return ref entries[idx];
            }
        }
    }
#pragma warning restore CS9181 

    /// <summary>
    /// Clears all entries in the cluster by setting their depth to 0.
    /// Entries with depth 0 are considered empty and available for use.
    /// </summary>
    public void Clear()
    {
        for (var i = 0; i < NumEntries; i++)
            this[i].Depth = 0;
    }
}

internal class TranspositionTable
{
    const int GenInc = 1;

    readonly TTCluster[] _table;
    public byte Generation { get; private set; }
    public long Size { get; private set; }

    /// <summary>
    /// Initializes a new transposition table with the specified memory size.
    /// The actual size will be rounded down to the nearest power of 2 for efficient indexing.
    /// </summary>
    /// <param name="sizeBytes">The desired size of the transposition table in bytes</param>
    public TranspositionTable(long sizeBytes)
    {
        var numClusters = sizeBytes / Marshal.SizeOf<TTCluster>();
        Size = 1L << MathFunctions.FloorLog2((ulong)numClusters);
        _table = new TTCluster[numClusters];
        Generation = 0;
    }

    /// <summary>
    /// Clears all entries in the transposition table by resetting each cluster.
    /// </summary>
    public void Clear()
    {
        for (var i = 0; i < _table.Length; i++)
            _table[i].Clear();
    }

    /// <summary>
    /// Increments the generation number to age out old entries from previous searches.
    /// Called when the search root position changes.
    /// </summary>
    public void IncGen() => Generation += GenInc;

    /// <summary>
    /// Saves a transposition table entry without a best move (evaluation bounds only).
    /// </summary>
    /// <param name="entry">Reference to the entry to update</param>
    /// <param name="pos">The position being saved</param>
    /// <param name="lower">The lower bound (alpha) of the evaluation</param>
    /// <param name="upper">The upper bound (beta) of the evaluation</param>
    /// <param name="depth">The search depth for this evaluation</param>
    /// <param name="nodeCount">The number of nodes searched</param>
    public void SaveAt(ref TTEntry entry, ref Position pos, int lower, int upper, int depth, long nodeCount)
    => SaveAt(ref entry, ref pos, BoardCoordinate.Null, lower, upper, depth, nodeCount);

    /// <summary>
    /// Saves a complete transposition table entry with position, evaluation bounds, and best move.
    /// </summary>
    /// <param name="entry">Reference to the entry to update</param>
    /// <param name="pos">The position being saved</param>
    /// <param name="move">The best move found for this position</param>
    /// <param name="lower">The lower bound (alpha) of the evaluation</param>
    /// <param name="upper">The upper bound (beta) of the evaluation</param>
    /// <param name="depth">The search depth for this evaluation</param>
    /// <param name="nodeCount">The number of nodes searched (used for replacement policy)</param>
    public void SaveAt(ref TTEntry entry, ref Position pos, BoardCoordinate move, int lower, int upper, int depth, long nodeCount)
    {
        entry.Position = pos.GetBitboard();
        entry.Move = move;
        entry.Lower = (short)lower;
        entry.Upper = (short)upper;
        entry.Generation = Generation;
        entry.Cost = (byte)MathFunctions.FloorLog2((ulong)nodeCount);
        entry.Depth = (byte)depth;
    }

    /// <summary>
    /// Retrieves or allocates a transposition table entry for the given position.
    /// Uses 4-way set-associative lookup with a replacement policy based on priority.
    /// </summary>
    /// <param name="pos">The position to look up</param>
    /// <param name="hit">Output parameter indicating whether the position was found (true) or a new entry was allocated (false)</param>
    /// <returns>A reference to the transposition table entry for this position</returns>
    public ref TTEntry GetEntry(ref Position pos, out bool hit)
    {
        // Equivalent to: idx = pos.hash_code() % size
        // Since size is a power of 2, we can use bitwise AND for efficient modulo
        long idx = (long)pos.ComputeHashCode() & (Size - 1);

        Debug.Assert(idx < Size);

        ref TTCluster entries = ref _table[idx];

        for (int i = 0; i < TTCluster.NumEntries; i++)
        {
            ref TTEntry entry = ref entries[i];

            // Empty entry found - initialize and return it
            if (entry.Depth == 0)
            {
                hit = false;
                entries[i].Lower = -Searcher.ValueInf;
                entries[i].Upper = Searcher.ValueInf;
                return ref entries[i];
            }

            // Position match found!
            if (pos.Has(ref entries[i].Position))
            {
                hit = true;
                return ref entries[i];
            }
        }

        // Position not found, so find the entry with lowest priority to replace
        ref TTEntry replace = ref entries[0];
        var maxPriority = replace.Priority;
        for (int i = 0; i < TTCluster.NumEntries; i++)
        {
            if (entries[i].Priority < maxPriority)
            {
                replace = ref entries[i];
                maxPriority = replace.Priority;
            }
        }

        hit = false;
        return ref replace;
    }

    /// <summary>
    /// Reconstructs the principal variation (PV) from the transposition table.
    /// Recursively follows best moves stored in the TT to build the complete line of play.
    /// </summary>
    /// <param name="pos">The current position (will be modified during recursion)</param>
    /// <param name="pv">The list to append PV moves to</param>
    /// <param name="depth">The remaining depth to search for PV moves</param>
    public void ProbePV(ref Position pos, List<BoardCoordinate> pv, int depth)
    {
        ref TTEntry entry = ref GetEntry(ref pos, out var hit);

        if(hit && entry.Generation == Generation && entry.Depth == depth && entry.Move != BoardCoordinate.Null)
        {
            pv.Add(entry.Move);
            var move = new Move(entry.Move);
            pos.CalcFlip(ref move);
            
            if (pos.CanPass)
                pos.Pass();

            ProbePV(ref pos, pv, depth - 1);
            pos.Undo(ref move);
        }
    }
}