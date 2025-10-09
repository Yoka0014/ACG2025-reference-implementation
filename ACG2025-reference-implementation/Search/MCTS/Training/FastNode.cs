namespace ACG2025_reference_implementation.Search.MCTS.Training;

using System;
using System.Linq;
using System.Runtime.CompilerServices;

using ACG2025_reference_implementation.NTupleSystem;
using ACG2025_reference_implementation.Reversi;
using ACG2025_reference_implementation.Search;
using ACG2025_reference_implementation.Search.MCTS;

/// <summary>
/// A lightweight, performance-optimized version of the standard Node class specifically designed for training scenarios.
/// Unlike the regular Node class, FastNode uses fixed-size arrays to avoid dynamic memory allocation during MCTS simulations,
/// making it more suitable for high-frequency training operations where performance is critical.
/// </summary>
/// <param name="nTupleManager">The n-tuple manager for feature extraction and evaluation</param>
internal class FastNode(NTupleManager nTupleManager)
{
    /// <summary>The game state associated with this node, containing position and feature information</summary>
    public State State = new(nTupleManager);
    
    /// <summary>Number of times this node has been visited during MCTS simulations</summary>
    public int VisitCount;
    
    /// <summary>Number of child edges/nodes currently in use (may be less than maximum possible moves)</summary>
    public int NumChildren;

    /// <summary>Fixed-size array for storing edges to child nodes to avoid dynamic allocation</summary>
    readonly Edge[] _edges = new Edge[Constants.MaxLegalMoves];
    
    /// <summary>Fixed-size array for storing child node references to avoid dynamic allocation</summary>
    readonly FastNode[] _childNodes = new FastNode[Constants.MaxLegalMoves];

    /// <summary>Gets a span view of the currently active edges based on NumChildren</summary>
    public Span<Edge> Edges => _edges.AsSpan(0, NumChildren);
    
    /// <summary>Gets a span view of the currently active child nodes based on NumChildren</summary>
    public Span<FastNode> ChildNodes => _childNodes.AsSpan(0, NumChildren);

    /// <summary>
    /// Expands this node by creating edges for all legal moves from the current position.
    /// If no legal moves are available, creates a single pass move.
    /// </summary>
    /// <param name="moves">Span containing the legal moves available from this position</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Expand(Span<Move> moves)
    {
        if (moves.Length == 0)
        {
            _edges[0].Move.Coord = BoardCoordinate.Pass;
            NumChildren = 1;
            return;
        }

        for (var i = 0; i < moves.Length; i++)
            _edges[i].Move = moves[i];
        NumChildren = moves.Length;
    }

    /// <summary>
    /// Resets this node to its initial state by clearing all edges, child nodes, and statistics.
    /// This method is used when recycling nodes from the object pool to avoid memory allocations.
    /// </summary>
    public void Clear()
    {
        Array.Clear(_edges, 0, NumChildren);
        Array.Clear(_childNodes, 0, NumChildren);
        NumChildren = VisitCount = 0;
    }
}

/// <summary>
/// An object pool for managing FastNode instances to minimize garbage collection pressure during training.
/// This pool pre-allocates a fixed number of FastNode objects and reuses them throughout MCTS simulations,
/// which is more efficient than creating and destroying nodes dynamically.
/// </summary>
/// <param name="nTupleManager">The n-tuple manager for creating new FastNode instances</param>
/// <param name="size">The initial size of the object pool</param>
internal class NodePool(NTupleManager nTupleManager, int size)
{
    /// <summary>Pre-allocated array of FastNode objects for reuse</summary>
    readonly FastNode[] _nodes = [.. Enumerable.Range(0, size).Select(_ => new FastNode(nTupleManager))];
    
    /// <summary>Reference to the n-tuple manager for creating additional nodes if pool is exhausted</summary>
    readonly NTupleManager _nTupleManager = nTupleManager;
    
    /// <summary>Current position in the pool indicating the next available node</summary>
    int _loc = 0;

    /// <summary>
    /// Retrieves a cleared FastNode from the pool. If the pool is exhausted,
    /// creates a new FastNode instance dynamically.
    /// </summary>
    /// <returns>A clean FastNode ready for use</returns>
    public FastNode Get()
    {
        if (_loc == _nodes.Length)
            return new FastNode(_nTupleManager);

        var node = _nodes[_loc++];
        node.Clear();
        return node;
    }

    /// <summary>
    /// Resets the pool to its initial state, making all pre-allocated nodes available for reuse.
    /// This should be called at the beginning of each search iteration.
    /// </summary>
    public void Clear() => _loc = 0;
}