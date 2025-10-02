namespace ACG2025_reference_implementation.Search.AlphaBetaPruning;

using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using ACG2025_reference_implementation.Utils;
using ACG2025_reference_implementation.Reversi;

internal struct TTEntry
{
    public Bitboard Position;
    public short Lower;
    public short Upper;
    public BoardCoordinate Move;
    public byte Generation;
    public byte Cost;
    public byte Depth;

    public readonly int Priority => (Generation << 16) | (Cost << 8) | Depth;
    public readonly bool HasBestMove => this.Lower == this.Upper;

    public TTEntry() { }
}

[InlineArray(NumEntries)]
internal struct TTCluster
{
    public const int NumEntries = 4;

    TTEntry _entry;

#pragma warning disable CS9181 // アクセスの高速化のために独自のインデクサを用いるので警告を抑制する．
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

    public TranspositionTable(long sizeBytes)
    {
        var numClusters = sizeBytes / Marshal.SizeOf<TTCluster>();
        Size = 1L << MathFunctions.FloorLog2((ulong)numClusters);
        _table = new TTCluster[numClusters];
        Generation = 0;
    }

    public void Clear()
    {
        for (var i = 0; i < _table.Length; i++)
            _table[i].Clear();
    }

    public void IncGen() => Generation += GenInc;

    public void SaveAt(ref TTEntry entry, ref Position pos, int lower, int upper, int depth, long nodeCount)
    => SaveAt(ref entry, ref pos, BoardCoordinate.Null, lower, upper, depth, nodeCount);

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

    public ref TTEntry GetEntry(ref Position pos, out bool hit)
    {
        // idx = pos.hash_code() % sizeと同じ．
        // sizeが2^nとなるようにしているため，以下のように計算できる．
        long idx = (long)pos.ComputeHashCode() & (Size - 1);

        Debug.Assert(idx < Size);

        ref TTCluster entries = ref _table[idx];

        for (int i = 0; i < TTCluster.NumEntries; i++)
        {
            ref TTEntry entry = ref entries[i];

            // 空エントリ
            if (entry.Depth == 0)
            {
                hit = false;
                entries[i].Lower = -Searcher.ValueInf;
                entries[i].Upper = Searcher.ValueInf;
                return ref entries[i];
            }

            // found it!
            if (pos.Has(ref entries[i].Position))
            {
                hit = true;
                return ref entries[i];
            }
        }

        // キーで指定された局面が見つからなかったので上書きするエントリを返す．
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