namespace ACG2025_reference_implementation.Search.MCTS.Training;

using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

using MathNet.Numerics.Distributions;

using ACG2025_reference_implementation.Utils;
using ACG2025_reference_implementation.Reversi;
using ACG2025_reference_implementation.Evaluation;
using ACG2025_reference_implementation.Search.MCTS;

using static PUCTConstants;

/// <summary>
/// A streamlined, single-threaded implementation of PUCT search optimized for training scenarios.
/// Unlike the full-featured PUCTSearcher, this version focuses on performance and simplicity,
/// removing parallel processing, event handling, and complex search management features.
/// It is specifically designed for self-play training where many fast simulations are needed.
/// </summary>
/// <param name="valueFunc">The value function used for position evaluation and move prioritization</param>
/// <param name="numSimulations">The number of MCTS simulations to perform per search</param>
internal class FastPUCTSearcher(ValueFunction valueFunc, int numSimulations)
{
    /// <summary>Gets or sets the Dirichlet alpha parameter for root node exploration noise</summary>
    public double RootDirchletAlpha { get; set; } = 0.3;

    /// <summary>Gets or sets the fraction of root prior probability to replace with Dirichlet noise</summary>
    public double RootExplorationFraction { get; set; } = 0.25;

    /// <summary>The fixed number of MCTS simulations to perform during each search</summary>
    readonly int _numSimulations = numSimulations;

    /// <summary>Value function used for position evaluation and move prior probability calculation</summary>
    readonly ValueFunction _valueFunc = valueFunc;

    /// <summary>Object pool for managing FastNode instances to reduce garbage collection overhead</summary>
    readonly NodePool _nodePool = new(valueFunc.NTupleManager, (int)(numSimulations * 1.5));

    /// <summary>Root node of the current search tree, or null if no search tree exists</summary>
    FastNode? _root;

    /// <summary>Proof status of the root position (win/loss/draw/not proved)</summary>
    EdgeLabel _rootEdgeLabel;

    /// <summary>Game state corresponding to the root position of the search tree</summary>
    State _rootState;

    /// <summary>
    /// Gets the root value representing the expected reward for the current player at the root position.
    /// Returns the proven outcome value if the position is solved, otherwise returns the expected reward from MCTS evaluation.
    /// Returns NaN if no root node exists.
    /// </summary>
    public double RootValue
    {
        get
        {
            if (_root is null)
                return double.NaN;

            if (_rootEdgeLabel != EdgeLabel.NotProved)
                return 1.0 - OutcomeToReward[(int)(_rootEdgeLabel ^ EdgeLabel.Proved)];

            var valueSum = 0.0;
            for (var i = 0; i < _root.Edges.Length; i++)
            {
                ref var edge = ref _root.Edges[i]; 
                if (edge.VisitCount != 0)
                    valueSum += edge.ExpectedReward;
            }

            return valueSum / _root.NumChildren;
        }
    }

    /// <summary>
    /// Sets the root position for the search tree and initializes a new search tree.
    /// This method prepares the searcher for a new position but doesn't start the search.
    /// </summary>
    /// <param name="pos">The game position to set as the root of the search tree</param>
    public void SetRootPosition(ref Position pos)
    {
        _rootState = new State(pos, _valueFunc.NTupleManager);
        _rootEdgeLabel = EdgeLabel.NotProved;
    }

    public void UpdateRootPosition(ref Move move)
    {
        _rootState.Update(ref move);
        _rootEdgeLabel = EdgeLabel.NotProved;
    }

    public void PassRootPosition()
    {
        _rootState.Pass();
        _rootEdgeLabel = EdgeLabel.NotProved;
    }

    /// <summary>
    /// Performs the complete MCTS search by running the specified number of simulations.
    /// This method initializes the search tree, sets up the root node with exploration noise,
    /// and performs all simulations in a single-threaded manner for training efficiency.
    /// </summary>
    public void Search()
    {
        _nodePool.Clear();
        _root = _nodePool.Get();
        _rootState.CopyTo(ref _root.State);
        InitRootChildNodes();

        for (var i = 0; i < _numSimulations; i++)
            VisitRootNode();
    }

    /// <summary>
    /// Selects the best move from the current search tree based on visit count and expected reward.
    /// Prioritizes proven wins, avoids proven losses, and uses visit statistics for unproven moves.
    /// </summary>
    /// <returns>The best move, or null if no search tree exists</returns>
    public Move? SelectBestMove() => _root?.Edges[SelectBestChildNode(_root)].Move;

    /// <summary>
    /// Selects a move randomly according to the visit count distribution using the shared Random instance.
    /// Moves with higher visit counts have higher probability of being selected.
    /// This is commonly used in training to add stochasticity for exploration.
    /// </summary>
    /// <returns>The selected move, or null if no search tree exists</returns>
    public Move? SelectMoveWithVisitCountDist() => SelectMoveWithVisitCountDist(Random.Shared);

    /// <summary>
    /// Selects a move randomly according to the visit count distribution using the provided Random instance.
    /// The probability of selecting each move is proportional to its visit count during the search.
    /// This method is useful for training scenarios where exploration diversity is needed.
    /// </summary>
    /// <param name="rand">Random number generator to use for stochastic move selection</param>
    /// <returns>The selected move, or null if no search tree exists</returns>
    public Move? SelectMoveWithVisitCountDist(Random rand)
    {
        if (_root is null)
            return null;

        var edges = _root.Edges;
        Span<double> prob = stackalloc double[edges.Length];
        for (var i = 0; i < prob.Length; i++)
            prob[i] = edges[i].VisitCount;

        return edges[rand.Sample(prob)].Move;
    }

    /// <summary>
    /// Initializes the root node by expanding it with all legal moves, setting prior probabilities and values,
    /// and adding Dirichlet noise to encourage exploration. This method also pre-creates all child nodes
    /// to avoid allocation overhead during simulation.
    /// </summary>
    void InitRootChildNodes()
    {
        Span<Move> moves = stackalloc Move[Constants.MaxLegalMoves];
        moves = moves[.._root!.State.Position.GetNextMoves(ref moves)];
        _root.Expand(moves);

        if (moves.Length != 0)
            SetPriorProbsAndValues(_root);

        // Add Dirichlet noise to root prior probabilities to encourage exploration
        var edges = _root.Edges;
        var frac = RootExplorationFraction;
        var noise = Dirichlet.Sample(Random.Shared, [.. Enumerable.Repeat(RootDirchletAlpha, edges.Length)]);
        for (var i = 0; i < edges.Length; i++)
            edges[i].PriorProb = (Half)((double)edges[i].PriorProb * (1.0 - frac) + noise[i] * frac);

        // Pre-create all child nodes to avoid allocations during simulation
        for (var i = 0; i < _root.ChildNodes.Length; i++)
            CreateChildNode(_root, i);
    }

    /// <summary>
    /// Performs a single MCTS simulation starting from the root node.
    /// Selects a child using the PUCT formula, and either uses the initial value estimate
    /// for first-time visits or recursively visits the child node for subsequent visits.
    /// </summary>
    void VisitRootNode()
    {
        var edges = _root!.Edges;

        int childIdx;
        bool isFirstVisit;
        childIdx = SelectChildNode(_root, ref _rootEdgeLabel);
        isFirstVisit = edges[childIdx].VisitCount == 0;

        ref var childEdge = ref edges[childIdx];
        if (isFirstVisit)
            UpdateNodeStats(_root, ref childEdge, (double)childEdge.Value);
        else
            UpdateNodeStats(_root, ref childEdge, VisitNode(_root.ChildNodes[childIdx], ref childEdge));
    }

    /// <summary>
    /// Recursively visits a node in the MCTS tree, handling expansion, move selection, and backpropagation.
    /// This method manages both regular moves and pass moves, and handles terminal game positions.
    /// </summary>
    /// <param name="node">The node to visit</param>
    /// <param name="edgeToNode">Reference to the edge leading to this node (for proof propagation)</param>
    /// <param name="afterPass">True if this visit follows a pass move, indicating potential game end</param>
    /// <returns>The value to backpropagate (from the current player's perspective)</returns>
    double VisitNode(FastNode node, ref Edge edgeToNode, bool afterPass = false)
    {
        var state = node.State;
        Span<Edge> edges;
        if (node.NumChildren == 0) // need to expand
        {
            Span<Move> moves = stackalloc Move[Constants.MaxLegalMoves];
            moves = moves[..state.Position.GetNextMoves(ref moves)];
            node.Expand(moves);
            edges = node.Edges;

            if (moves.Length != 0)
                SetPriorProbsAndValues(node);
        }
        else
        {
            edges = node.Edges;
        }

        double value;
        if (edges[0].Move.Coord == BoardCoordinate.Pass)
        {
            if (afterPass)  // gameover
            {
                var outcome = GetOutcome(ref state);
                edges[0].Label = EdgeLabel.Proved | (EdgeLabel)outcome;
                edgeToNode.Label = EdgeLabel.Proved | (EdgeLabel)ToOpponentOutcome[outcome];

                value = OutcomeToReward[outcome];
            }
            else if (edges[0].IsProved)
            {
                value = OutcomeToReward[(int)(edges[0].Label ^ EdgeLabel.Proved)];
            }
            else
            {
                var childNode = node.ChildNodes[0] ?? CreatePassChildNode(node);
                value = VisitNode(childNode, ref edges[0], afterPass: true);

                if (edges[0].IsProved)
                {
                    var outcome = (int)(edges[0].Label ^ EdgeLabel.Proved);
                    edgeToNode.Label = EdgeLabel.Proved | (EdgeLabel)ToOpponentOutcome[outcome];
                }
            }

            UpdateNodeStats(node, ref edges[0], value);
            return 1.0 - value;
        }

        // Handle non-pass moves
        var childIdx = SelectChildNode(node, ref edgeToNode.Label);
        ref var childEdge = ref edges[childIdx];
        var isFirstVisit = childEdge.VisitCount == 0;

        if (isFirstVisit)
        {
            value = (double)childEdge.Value;
        }
        else if (childEdge.IsProved)
        {
            value = OutcomeToReward[(int)(childEdge.Label ^ EdgeLabel.Proved)];
        }
        else
        {
            var childNodes = node.ChildNodes;
            var childNode = childNodes[childIdx] ?? CreateChildNode(node, childIdx);
            value = VisitNode(childNode, ref childEdge);
        }

        UpdateNodeStats(node, ref childEdge, value);
        return 1.0 - value;
    }

    /// <summary>
    /// Creates a child node for a regular move by getting a node from the pool and setting up its state.
    /// The parent state is temporarily updated to create the child state, then restored.
    /// </summary>
    /// <param name="node">The parent node</param>
    /// <param name="idx">The index of the child to create</param>
    /// <returns>The newly created child node</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    FastNode CreateChildNode(FastNode node, int idx)
    {
        ref var state = ref node.State;
        var child = node.ChildNodes[idx] = _nodePool.Get();
        ref var move = ref node.Edges[idx].Move;
        state.Update(ref move);
        state.CopyTo(ref child.State);
        state.Undo(ref move);
        return child;
    }

    /// <summary>
    /// Creates a child node for a pass move by getting a node from the pool and setting up its state.
    /// The parent state is temporarily passed to create the child state, then restored.
    /// </summary>
    /// <param name="node">The parent node</param>
    /// <returns>The newly created child node for the pass move</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    FastNode CreatePassChildNode(FastNode node)
    {
        ref var state = ref node.State;
        var child = node.ChildNodes[0] = _nodePool.Get();
        state.Pass();
        state.CopyTo(ref child.State);
        state.Pass();
        return child;
    }

    /// <summary>
    /// Sets the prior probabilities and initial value estimates for all edges of a node using the value function.
    /// Applies softmax normalization to the value estimates to create a probability distribution over moves.
    /// This method is highly optimized with aggressive compilation flags for performance-critical training scenarios.
    /// </summary>
    /// <param name="node">The node whose edges need prior probabilities and values set</param>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    unsafe void SetPriorProbsAndValues(FastNode node)
    {
        ref var state = ref node.State;
        var edges = node.Edges;
        float value;
        float expValueSum = 0.0f;
        var expValues = stackalloc float[edges.Length];
        for (var i = 0; i < edges.Length; i++)
        {
            ref var edge = ref edges[i];
            ref Move move = ref edge.Move;
            state.Position.CalcFlip(ref move);
            state.Update(ref edge.Move);
            edge.Value = (Half)(value = 1 - _valueFunc.PredictWinRate<float>(state.FeatureVector));
            expValueSum += expValues[i] = FastMath.Exp(value);
            state.Undo(ref edge.Move);
        }

        // Apply softmax normalization to convert exponential values to probabilities.
        for (var i = 0; i < edges.Length; i++)
            edges[i].PriorProb = (Half)(expValues[i] / expValueSum);
    }

    /// <summary>
    /// Selects the best child node for final move selection (not exploration during search).
    /// Uses a deterministic strategy: prioritizes proven wins, avoids proven losses,
    /// and selects based on visit count and expected reward for unproven moves.
    /// </summary>
    /// <param name="parent">Parent node to select the best child from</param>
    /// <returns>Index of the best child node for move selection</returns>
    static int SelectBestChildNode(FastNode parent)
    {
        var edges = parent.Edges;
        var maxIdx = 0;

        for (var i = 0; i < edges.Length; i++)
        {
            ref var edge = ref edges[i];

            if (edge.IsWin)
                return i;

            if (edge.IsLoss)
                continue;

            if (edge.PriorTo(ref edges[maxIdx]))
                maxIdx = i;
        }

        return maxIdx;
    }

    /// <summary>
    /// Selects the best child node using the PUCT (Predictor + Upper Confidence Bounds applied to Trees) selection formula.
    /// Balances exploitation (Q-value) and exploration (prior probability and visit count).
    /// Also handles proved positions (win/loss/draw) and propagates proof information to parent nodes.
    /// </summary>
    /// <param name="parent">Parent node to select child from during MCTS simulation</param>
    /// <param name="parentEdgeLabel">Output parameter that receives the parent's proof status after selection</param>
    /// <returns>Index of the selected child for the next simulation step</returns>
    static int SelectChildNode(FastNode parent, ref EdgeLabel parentEdgeLabel)
    {
        var edges = parent.Edges;
        var maxIdx = 0;
        var maxScore = float.NegativeInfinity;
        var visitSum = parent.VisitCount;
        var sqrtVisitSum = MathF.Sqrt(visitSum + Epsilon);

        var drawCount = 0;
        var lossCount = 0;
        for (var i = 0; i < edges.Length; i++)
        {
            ref var edge = ref edges[i];

            if (edge.IsWin)
            {
                // If there is a winning edge from the current player's view, it means a loss for the opponent.
                parentEdgeLabel = EdgeLabel.Loss;
                return i;
            }

            if (edge.IsLoss)
            {
                lossCount++;
                continue;   // avoid to select loss edge.
            }

            if (edge.IsDraw)
                drawCount++;

            // Calculate PUCT score (Q-value + exploration bonus).
            var q = (float)(edge.ValueSum / (edge.VisitCount + Epsilon));
            var u = PUCTFactor * (float)edge.PriorProb * sqrtVisitSum / (1.0f + edge.VisitCount);
            var score = q + u;

            if (score > maxScore)
            {
                maxScore = score;
                maxIdx = i;
            }
        }

        if (lossCount + drawCount == edges.Length)
            parentEdgeLabel = (drawCount != 0) ? EdgeLabel.Draw : EdgeLabel.Win;

        return maxIdx;
    }

    /// <summary>
    /// Updates the visit statistics for both a parent node and a child edge with the backpropagated value.
    /// This method is called during the backpropagation phase of MCTS to accumulate visit counts and value sums.
    /// </summary>
    /// <param name="parent">The parent node whose visit count should be incremented</param>
    /// <param name="childEdge">The child edge whose statistics should be updated</param>
    /// <param name="value">The value to add to the edge's value sum (from current player's perspective)</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void UpdateNodeStats(FastNode parent, ref Edge childEdge, double value)
    {
        parent.VisitCount++;
        childEdge.VisitCount++;
        childEdge.ValueSum += value;
    }

    /// <summary>
    /// Determines the game outcome from a terminal position by examining the disc difference.
    /// Used when both players have passed and the game has ended.
    /// </summary>
    /// <param name="state">Terminal game state to evaluate</param>
    /// <returns>Outcome flag: OutcomeWin if positive disc difference, OutcomeLoss if negative, OutcomeDraw if zero</returns>
    static int GetOutcome(ref State state)
    {
        var score = state.Position.DiscDiff;
        if (score == 0)
            return OutcomeDraw;
        return score > 0 ? OutcomeWin : OutcomeLoss;
    }
}