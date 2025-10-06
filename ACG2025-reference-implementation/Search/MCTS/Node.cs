namespace ACG2025_reference_implementation.Search.MCTS;

using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;

using ACG2025_reference_implementation.Reversi;

/// <summary>
/// Represents the proof status of an edge in the MCTS tree.
/// Used to track proven game outcomes (win/loss/draw) for game tree nodes.
/// </summary>
internal enum EdgeLabel
{
    /// <summary>The edge outcome is not yet proved</summary>
    NotProved = 0x00,
    
    /// <summary>Base flag indicating the edge outcome is proved</summary>
    Proved = 0xf0,
    
    /// <summary>Proved winning edge</summary>
    Win = Proved | PUCTConstants.OutcomeWin,
    
    /// <summary>Proved losing edge</summary>
    Loss = Proved | PUCTConstants.OutcomeLoss,
    
    /// <summary>Proved draw edge</summary>
    Draw = Proved | PUCTConstants.OutcomeDraw
}

/// <summary>
/// Represents an edge in the MCTS tree connecting a parent node to a child node.
/// Contains move information, statistics, and proof status for the edge.
/// </summary>
internal struct Edge
{
    /// <summary>The move that this edge represents</summary>
    public Move Move;
    
    /// <summary>The prior probability of selecting this move (derived from value function evaluation)</summary>
    public Half PriorProb;
    
    /// <summary>The initial value estimate for this move</summary>
    public Half Value;
    
    /// <summary>Number of times this edge has been visited during search</summary>
    public uint VisitCount;
    
    /// <summary>Sum of all reward values propagated through this edge</summary>
    public double ValueSum;
    
    /// <summary>Proof status of this edge (not proved, win, loss, or draw)</summary>
    public EdgeLabel Label;

    /// <summary>Gets the average reward (Q-value) for this edge</summary>
    public readonly double ExpectedReward => ValueSum / VisitCount;
    
    /// <summary>Gets whether this edge has a proved outcome</summary>
    public readonly bool IsProved => (Label & EdgeLabel.Proved) != 0;
    
    /// <summary>Gets whether this edge is proved to be a winning move</summary>
    public readonly bool IsWin => Label == EdgeLabel.Win;
    
    /// <summary>Gets whether this edge is proved to be a losing move</summary>
    public readonly bool IsLoss => Label == EdgeLabel.Loss;
    
    /// <summary>Gets whether this edge is proved to result in a draw</summary>
    public readonly bool IsDraw => Label == EdgeLabel.Draw;

    /// <summary>
    /// Determines if this edge has higher priority than another edge for move selection.
    /// Prioritizes edges with higher visit counts, and breaks ties using expected reward.
    /// </summary>
    /// <param name="edge">The edge to compare against</param>
    /// <returns>True if this edge has higher priority, false otherwise</returns>
    public readonly bool PriorTo(ref Edge edge)
    {
        if (VisitCount == 0)
            return false;

        var diff = (long)VisitCount - edge.VisitCount;
        if (diff != 0)
            return diff > 0;
        return ExpectedReward > edge.ExpectedReward;
    }
}

/// <summary>
/// Represents a node in the Monte Carlo Tree Search (MCTS) tree.
/// Each node corresponds to a game position and contains edges to possible child positions.
/// </summary>
internal class Node
{
    /// <summary>Total number of times this node has been visited during search</summary>
    public uint VisitCount;
    
    /// <summary>Array of edges representing all legal moves from this position</summary>
    public Edge[]? Edges;
    
    /// <summary>Array of child nodes corresponding to positions after each legal move</summary>
    public Node[]? ChildNodes;

    /// <summary>Gets whether this node has been expanded (edges have been generated)</summary>
    public bool IsExpanded => Edges is not null;
    
    /// <summary>Gets whether child nodes array has been initialized</summary>
    public bool ChildNodeWasInitialized => ChildNodes is not null;

    /// <summary>
    /// Gets the expected reward (average Q-value) for this node based on all child edges.
    /// Returns NaN if the node is not expanded.
    /// </summary>
    public double ExpectedReward
    {
        get
        {
            if (Edges is null)
                return double.NaN;

            var reward = 0.0;
            for (var i = 0; i < Edges.Length; i++)
            {
                Debug.Assert(VisitCount != 0);
                reward += Edges[i].ValueSum / VisitCount;
            }
            return reward;
        }
    }

    /// <summary>
    /// Creates and initializes a child node at the specified index.
    /// </summary>
    /// <param name="idx">Index of the child node to create</param>
    /// <returns>The newly created Node object</returns>
    public Node CreateChildNode(int idx)
    {
        Debug.Assert(ChildNodes is not null);
        return ChildNodes[idx] = new Node();
    }

    /// <summary>
    /// Creates a Node array with the same length as the number of legal moves.
    /// All elements are initialized to null at this point, and Node objects are created
    /// using the CreateChildNode method when needed.
    /// </summary>
    /// <returns>Array of child nodes initialized to null</returns>
    public Node[] InitChildNodes()
    {
        Debug.Assert(Edges is not null);
        return ChildNodes = new Node[Edges.Length];
    }

    /// <summary>
    /// Expands this node by generating edges for all legal moves from the current position.
    /// If no legal moves are available, creates a single pass move edge.
    /// </summary>
    /// <param name="pos">The game position corresponding to this node</param>
    /// <returns>Array of edges representing all legal moves</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Edge[] Expand(ref Position pos)
    {
        Span<Move> moves = stackalloc Move[Constants.MaxLegalMoves];
        var numMoves = pos.GetNextMoves(ref moves);

        if (numMoves == 0)
        {
            Edges = new Edge[1];
            Edges[0].Move.Coord = BoardCoordinate.Pass;
            return Edges;
        }

        Edges = new Edge[numMoves];
        for (var i = 0; i < Edges.Length; i++)
            Edges[i].Move = moves[i];

        return Edges;
    }
}