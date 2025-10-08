namespace ACG2025_reference_implementation.Search.MCTS;

using System;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using System.Runtime.CompilerServices;

using MathNet.Numerics.Distributions;

using ACG2025_reference_implementation.Utils;
using ACG2025_reference_implementation.Reversi;
using ACG2025_reference_implementation.Evaluation;

using static PUCTConstants;

/// <summary>
/// Constants used in the PUCT (Predictor + Upper Confidence Bounds applied to Trees) search algorithm.
/// These constants define exploration parameters and outcome mappings for the MCTS implementation.
/// </summary>
internal static class PUCTConstants
{
    /// <summary>Exploration factor in the PUCT formula</summary>
    public const float PUCTFactor = 1.0f;

    /// <summary>Virtual loss added during parallel search to avoid overexploring the same path</summary>
    public const uint VirtualLoss = 1;

    /// <summary>Bit flag representing a winning outcome</summary>
    public const int OutcomeWin = 1;

    /// <summary>Bit flag representing a losing outcome</summary>
    public const int OutcomeLoss = 1 << 1;

    /// <summary>Bit flag representing a draw outcome</summary>
    public const int OutcomeDraw = 1 << 2;

    /// <summary>Maps outcome flags to reward values (Win=1.0, Loss=0.0, Draw=0.5)</summary>
    public static ReadOnlySpan<float> OutcomeToReward => [float.NaN, 1.0f, 0.0f, float.NaN, 0.5f];

    /// <summary>Maps outcome from current player's perspective to opponent's perspective</summary>
    public static ReadOnlySpan<int> ToOpponentOutcome => [0, OutcomeLoss, OutcomeWin, 0, OutcomeDraw];
}

/// <summary>
/// Represents the evaluation result for a move in the search tree.
/// Contains move information, search statistics, and the principal variation.
/// </summary>
/// <param name="pv">Principal variation (sequence of best moves) from this position</param>
internal class MoveEval(IEnumerable<BoardCoordinate> pv) : IComparable<MoveEval>
{
    /// <summary>The move being evaluated</summary>
    public BoardCoordinate Move { get; init; }

    /// <summary>Proportion of search effort spent on this move (0.0 to 1.0)</summary>
    public double Effort { get; init; }

    /// <summary>Number of simulations performed for this move</summary>
    public long SimulationCount { get; init; }

    /// <summary>Expected reward (Q-value) for this move</summary>
    public double ExpectedReward { get; init; }

    /// <summary>Proven game result for this move, if any</summary>
    public GameResult GameResult { get; init; }

    /// <summary>Gets the principal variation (best line of play) from this move</summary>
    public ReadOnlySpan<BoardCoordinate> PV => _pv;

    readonly BoardCoordinate[] _pv = [.. pv];

    /// <summary>
    /// Determines if this move evaluation has higher priority than another.
    /// Prioritizes proven results (Win > Draw > Loss), then simulation count, then expected reward.
    /// </summary>
    /// <param name="other">Move evaluation to compare against</param>
    /// <returns>True if this move has higher priority, false otherwise</returns>
    public bool PriorTo(MoveEval other)
    {
        if (GameResult != GameResult.NotOver)
        {
            if (GameResult == GameResult.Win)
                return true;

            if (GameResult == GameResult.Loss)
                return false;

            if (GameResult == GameResult.Draw)
            {
                if (other.GameResult == GameResult.Loss || other.GameResult == GameResult.Draw)
                    return true;
            }
        }

        var diff = SimulationCount - other.SimulationCount;
        if (diff != 0)
            return diff > 0;
        return ExpectedReward > other.ExpectedReward;
    }

    /// <summary>
    /// Compares this move evaluation with another for sorting purposes.
    /// Used by Array.Sort and similar methods to order moves by priority.
    /// </summary>
    /// <param name="other">Move evaluation to compare against</param>
    /// <returns>Comparison result: negative if this is less, positive if greater, zero if equal</returns>
    public int CompareTo(MoveEval? other)
    {
        if (other is null)
            return 1;

        static int GameResultToPriority(GameResult res)
        => res switch
        {
            GameResult.Win => 3,
            GameResult.Draw => 2,
            GameResult.Loss => 1,
            GameResult.NotOver => 0,
            _ => throw new NotImplementedException()
        };

        var priority = GameResultToPriority(GameResult);
        var otherPriority = GameResultToPriority(other.GameResult);

        if (priority != otherPriority)
            return priority.CompareTo(otherPriority);

        return SimulationCount.CompareTo(other.SimulationCount);
    }
}

/// <summary>
/// Contains the complete results of a MCTS, including evaluation of the root position
/// and all child moves sorted by priority.
/// </summary>
/// <param name="rootEval">Evaluation of the root position</param>
/// <param name="childEvals">Evaluations of all child moves</param>
internal class SearchResult(MoveEval rootEval, IEnumerable<MoveEval> childEvals)
{
    /// <summary>Evaluation result for the root position</summary>
    public MoveEval RootEval { get; } = rootEval;

    /// <summary>Evaluation results for all child moves, sorted by priority</summary>
    public ReadOnlySpan<MoveEval> ChildEvals => _childEvals;

    readonly MoveEval[] _childEvals = [.. childEvals];
}

/// <summary>
/// Implementation of PUCT (Predictor + Upper Confidence Bounds applied to Trees) search algorithm.
/// This is a Monte Carlo Tree Search variant that uses n-tuple-based value function's guidance for move selection
/// and position evaluation, with parallel tree search support.
/// </summary>
internal class PUCTSearcher
{
    const float Epsilon = 1.0e-6f;

    /// <summary>Event fired when search results are updated during search</summary>
    public event EventHandler<SearchResult?> SearchResultUpdated = delegate { };

    /// <summary>Gets or sets the interval for search result updates in centiseconds</summary>
    public int SearchResultUpdateIntervalCs { get; set; }

    /// <summary>Gets or sets the Dirichlet alpha parameter for root node exploration noise</summary>
    public double RootDirchletAlpha { get; set; } = 0.3;

    /// <summary>Gets or sets the fraction of root prior probability to replace with Dirichlet noise</summary>
    public double RootExplorationFraction { get; set; } = 0.25;

    /// <summary>Value function used for position evaluation</summary>
    readonly ValueFunction _valueFunc;

    /// <summary>Root node of the search tree</summary>
    Node? _root;

    /// <summary>Proof label for the root edge</summary>
    EdgeLabel _rootEdgeLabel;

    /// <summary>Game position at the root of the search tree</summary>
    Position _rootPos;

    /// <summary>Flag indicating whether search is currently running</summary>
    volatile bool _isSearching;

    /// <summary>Cancellation token source for stopping search</summary>
    CancellationTokenSource? _cts;

    /// <summary>Timestamp when search started (in milliseconds)</summary>
    int _searchStartTimeMs = 0;

    /// <summary>Timestamp when search ended (in milliseconds)</summary>
    int _searchEndTimeMs = 0;

    /// <summary>Number of threads to use for parallel search</summary>
    int _numThreads = Environment.ProcessorCount;

    /// <summary>Node count per thread for tracking search statistics</summary>
    long[] _nodeCountPerThread;

    /// <summary>Maximum number of simulations to perform</summary>
    long _maxSimulationCount;

    /// <summary>Current number of simulations performed</summary>
    long _simulationCount;

    /// <summary>
    /// Initializes a new instance of the PUCTSearcher class.
    /// </summary>
    /// <param name="valueFunc">Value function to use for position evaluation</param>
    public PUCTSearcher(ValueFunction valueFunc)
    {
        _valueFunc = valueFunc;
        _nodeCountPerThread = new long[_numThreads];
    }

    /// <summary>Gets whether a search is currently in progress</summary>
    public bool IsSearching => _isSearching;

    /// <summary>Gets the elapsed time of the current or last search in milliseconds</summary>
    public int SearchElapsedMs => _isSearching ? Environment.TickCount - _searchStartTimeMs : _searchEndTimeMs - _searchStartTimeMs;

    /// <summary>Gets the total number of nodes visited across all threads</summary>
    public long NodeCount => _nodeCountPerThread.Sum();

    /// <summary>Gets the nodes per second rate of the search</summary>
    public double Nps => NodeCount / (SearchElapsedMs * 1.0e-3);

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
                return OutcomeToReward[(int)(_rootEdgeLabel ^ EdgeLabel.Proved)];

            return _root.ExpectedReward;
        }
    }

    /// <summary>
    /// Gets or sets the number of threads to use for parallel search.
    /// Cannot be changed while a search is in progress.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when trying to set while searching</exception>
    public int NumThreads
    {
        get => _numThreads;

        set
        {
            if (_isSearching)
                throw new InvalidOperationException("Cannot set the number of threads while searching.");

            _numThreads = value;

            var nodeCount = NodeCount;
            _nodeCountPerThread = new long[_numThreads];
            _nodeCountPerThread[0] = nodeCount;
        }
    }

    /// <summary>
    /// Sends a signal to stop the current search operation.
    /// </summary>
    public void SendStopSearchSignal() => _cts?.Cancel();

    /// <summary>
    /// Sets the root position for the search tree and initializes a new search tree.
    /// </summary>
    /// <param name="pos">The position to set as the root</param>
    public void SetRootPosition(ref Position pos)
    {
        _rootPos = pos;
        _root = new Node();
        _rootEdgeLabel = EdgeLabel.NotProved;
        InitRootChildNodes();
        Array.Clear(_nodeCountPerThread);
    }

    /// <summary>
    /// Transitions the root of the search tree to a child state after the specified move.
    /// This allows reusing the search tree when a move is played.
    /// </summary>
    /// <param name="move">The move that was played</param>
    /// <returns>True if the transition was successful, false otherwise</returns>
    public bool TransitionRootStateToChildState(BoardCoordinate move)
    {
        if (_root is null || !_root.IsExpanded || !_root.ChildNodeWasInitialized)
            return false;

        Debug.Assert(_root.Edges is not null && _root.ChildNodes is not null);

        Edge[] edges = _root.Edges;
        for (var i = 0; i < edges.Length; i++)
        {
            if (move == edges[i].Move.Coord && _root.ChildNodes[i] is not null)
            {
                if (move != BoardCoordinate.Pass)
                    _rootPos.Update(ref edges[i].Move);
                else
                    _rootPos.Pass();

                _root = _root.ChildNodes[i];
                _rootEdgeLabel = edges[i].Label;
                InitRootChildNodes();
                Array.Clear(_nodeCountPerThread);
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Gets the current search results including root evaluation and all child move evaluations.
    /// </summary>
    /// <returns>Search result containing move evaluations, or null if no search tree exists</returns>
    public SearchResult? GetSearchResult()
    {
        if (_root is null || _root.Edges is null)
            return null;

        var rootEdgeLabel = _rootEdgeLabel;

        if ((rootEdgeLabel & EdgeLabel.Proved) != 0)
        {
            // rootEdgeLabel contains the outcome from the opponent's perspective, so convert to current player's perspective.
            var outcome = (int)(rootEdgeLabel ^ EdgeLabel.Proved);
            rootEdgeLabel = EdgeLabel.Proved | (EdgeLabel)ToOpponentOutcome[outcome];
        }

        Edge[] edges = _root.Edges;
        var childEvals = new MoveEval[_root.Edges.Length];

        var rootEval = new MoveEval(GetPV(_root))
        {
            Move = BoardCoordinate.Null,
            Effort = 1.0,
            SimulationCount = _root.VisitCount,
            GameResult = EdgeLabelToGameResult(rootEdgeLabel),
            ExpectedReward = (rootEdgeLabel == EdgeLabel.NotProved) ? _root.ExpectedReward : OutcomeToReward[(int)(rootEdgeLabel ^ EdgeLabel.Proved)]
        };

        for (var i = 0; i < edges.Length; i++)
        {
            ref var edge = ref edges[i];
            childEvals[i] = new MoveEval(GetPV(_root.ChildNodes?[i], edge.Move.Coord))
            {
                Move = edge.Move.Coord,
                Effort = (_root.VisitCount != 0) ? (double)edge.VisitCount / _root.VisitCount : 0.0,
                SimulationCount = edge.VisitCount,
                GameResult = EdgeLabelToGameResult(edge.Label),
                ExpectedReward = edge.IsProved ? OutcomeToReward[(int)(edge.Label ^ EdgeLabel.Proved)] : edge.ExpectedReward
            };
        }

        Array.Sort(childEvals, (x, y) => -x.CompareTo(y));

        return new SearchResult(rootEval, childEvals);
    }

    /// <summary>
    /// Selects the best move from the current search tree based on visit count and expected reward.
    /// </summary>
    /// <returns>The best move, or null if no search tree exists</returns>
    public Move? SelectBestMove() => _root?.Edges?[SelectBestChildNode(_root)].Move;

    /// <summary>
    /// Selects a move randomly according to the visit count distribution.
    /// Moves with higher visit counts have higher probability of being selected.
    /// </summary>
    /// <returns>The selected move, or null if no search tree exists</returns>
    public Move? SelectMoveWithVisitCountDist() => SelectMoveWithVisitCountDist(Random.Shared);

    /// <summary>
    /// Selects a move randomly according to the visit count distribution.
    /// Moves with higher visit counts have higher probability of being selected.
    /// </summary>
    /// <param name="rand">Random number generator to use</param>
    /// <returns>The selected move, or null if no search tree exists</returns>
    public Move? SelectMoveWithVisitCountDist(Random rand)
    {
        if (_root is null || _root.Edges is null)
            return null;

        var edges = _root.Edges;
        Span<double> prob = stackalloc double[edges.Length];
        for (var i = 0; i < prob.Length; i++)
            prob[i] = edges[i].VisitCount;

        return edges[rand.Sample(prob)].Move;
    }

    /// <summary>
    /// Performs asynchronous MCTS search with specified limits and calls completion callback.
    /// </summary>
    /// <param name="numSimulations">Maximum number of simulations to perform</param>
    /// <param name="timeLimitCs">Time limit in centiseconds</param>
    /// <param name="onCompleted">Callback to execute when search completes</param>
    /// <returns>Task representing the asynchronous search operation</returns>
    public async Task SearchAsync(long numSimulations, int timeLimitCs, Action onCompleted)
    {
        _cts = new CancellationTokenSource();
        _isSearching = true;

        await Task.Run(() =>
        {
            Search(numSimulations, timeLimitCs);
            onCompleted();
        }).ConfigureAwait(false);
    }

    /// <summary>
    /// Performs MCTS search with the specified simulation count and time limits.
    /// Uses multiple threads for parallel tree search.
    /// </summary>
    /// <param name="numSimulations">Maximum number of simulations to perform</param>
    /// <param name="timeLimitCs">Time limit in centiseconds</param>
    /// <exception cref="InvalidOperationException">Thrown if root state is not initialized</exception>
    public void Search(long numSimulations, int timeLimitCs = int.MaxValue)
    {
        if (_root is null)
            throw new InvalidOperationException("The root state was not initialized.");

        _isSearching = true;
        _cts ??= new CancellationTokenSource();
        _maxSimulationCount = numSimulations;
        _simulationCount = 0;
        _searchStartTimeMs = Environment.TickCount;

        var searchTasks = new Task[_numThreads];
        for (var i = 0; i < searchTasks.Length; i++)
        {
            var state = new State(_rootPos, _valueFunc.NTupleManager);
            var threadID = i;
            searchTasks[i] = Task.Run(() => SearchWorker(threadID, state, _cts.Token));
        }

        var timeLimitMs = timeLimitCs * 10;
        WaitForSearch(searchTasks, timeLimitMs);

        SearchResultUpdated(this, GetSearchResult());

        _isSearching = false;
        _searchEndTimeMs = Environment.TickCount;
        _cts = null;
    }

    /// <summary>
    /// Performs MCTS search on a single thread for the specified number of simulations.
    /// Useful for testing and debugging purposes.
    /// </summary>
    /// <param name="numSimulations">Number of simulations to perform</param>
    public void SearchOnSingleThread(uint numSimulations)
    {
        var rootState = new State(_rootPos, _valueFunc.NTupleManager);
        var state = new State(_valueFunc.NTupleManager);
        for (var i = 0u; i < numSimulations && _rootEdgeLabel == EdgeLabel.NotProved; i++)
        {
            rootState.CopyTo(ref state);
            VisitRootNode(0, ref state);
            _simulationCount++;
        }
    }

    /// <summary>
    /// Worker method that runs on each search thread to perform MCTS simulations.
    /// </summary>
    /// <param name="threadID">ID of the thread running this worker</param>
    /// <param name="rootState">Root state to start simulations from</param>
    /// <param name="ct">Cancellation token to stop the worker</param>
    void SearchWorker(int threadID, State rootState, CancellationToken ct)
    {
        var state = new State(rootState.FeatureVector.NTupleManager);
        while (!ct.IsCancellationRequested)
        {
            if (Interlocked.Increment(ref _simulationCount) > _maxSimulationCount)
            {
                Interlocked.Decrement(ref _simulationCount);
                return;
            }

            rootState.CopyTo(ref state);
            VisitRootNode(threadID, ref state);
        }
    }

    /// <summary>
    /// Waits for search to complete based on time limit and periodically updates search results.
    /// </summary>
    /// <param name="searchTasks">Array of search worker tasks</param>
    /// <param name="timeLimitMs">Time limit in milliseconds</param>
    void WaitForSearch(Task[] searchTasks, int timeLimitMs)
    {
        var lastCheckPointMs = Environment.TickCount;

        while (!CanStop(timeLimitMs))
        {
            var searchResultUpdateIntervalMs = SearchResultUpdateIntervalCs * 10;
            if (searchResultUpdateIntervalMs != 0 && Environment.TickCount - lastCheckPointMs >= searchResultUpdateIntervalMs)
            {
                SearchResultUpdated(this, GetSearchResult());
                lastCheckPointMs = Environment.TickCount;
            }

            Thread.Sleep(10);
        }

        _cts?.Cancel();
        Task.WaitAll(searchTasks);
    }

    /// <summary>
    /// Determines whether the search should stop based on various conditions.
    /// </summary>
    /// <param name="timeLimitMs">Time limit in milliseconds</param>
    /// <returns>True if search should stop, false otherwise</returns>
    bool CanStop(int timeLimitMs)
    {
        Debug.Assert(_cts is not null);

        return _cts.IsCancellationRequested
            || (_rootEdgeLabel & EdgeLabel.Proved) != 0
            || SearchElapsedMs >= timeLimitMs
            || _simulationCount >= _maxSimulationCount;
    }

    /// <summary>
    /// Initializes the child nodes of the root, expanding it if necessary and setting up
    /// prior probabilities and values using the value function.
    /// </summary>
    void InitRootChildNodes()
    {
        Debug.Assert(_root is not null);

        if (!_root.IsExpanded)
        {
            var state = new State(_rootPos, _valueFunc.NTupleManager);

            _root.Expand(ref state.Position);

            if (state.Position.GetNumNextMoves() == 0)
            {
                if (!_root.ChildNodeWasInitialized)
                    _root.InitChildNodes();

                if (_root.ChildNodes![0] is null)
                    _root.CreateChildNode(0);

                return;
            }

            Debug.Assert(_root.Edges is not null);

            if (_root.Edges[0].Move.Coord != BoardCoordinate.Pass)
                SetPriorProbsAndValues(ref state, _root.Edges);
        }

        Debug.Assert(_root.Edges is not null);

        if (_root.Edges[0].Move.Coord == BoardCoordinate.Pass)
            return;

        // Add Dirichlet noise to prior probabilities at the root to improve exploration.
        var edges = _root.Edges;
        var frac = RootExplorationFraction;
        var noise = Dirichlet.Sample(Random.Shared, [.. Enumerable.Repeat(RootDirchletAlpha, edges.Length)]);
        for (var i = 0; i < edges.Length; i++)
            edges[i].PriorProb = (Half)((double)edges[i].PriorProb * (1.0 - frac) + noise[i] * frac);

        if (!_root.ChildNodeWasInitialized)
            _root.InitChildNodes();

        for (var i = 0; i < _root.ChildNodes!.Length; i++)
        {
            if (_root.ChildNodes[i] is null)
                _root.CreateChildNode(i);
        }
    }

    /// <summary>
    /// Visits the root node to perform one MCTS simulation.
    /// Selects a child, adds virtual loss, and recursively visits or evaluates the position.
    /// </summary>
    /// <param name="threadID">ID of the thread performing the visit</param>
    /// <param name="state">Current game state</param>
    void VisitRootNode(int threadID, ref State state)
    {
        Debug.Assert(_root is not null);
        Debug.Assert(_root.Edges is not null);
        Debug.Assert(_root.ChildNodes is not null);

        Edge[] edges = _root.Edges;

        int childIdx;
        bool isFirstVisit;
        lock (_root)
        {
            childIdx = SelectChildNode(_root, ref _rootEdgeLabel);
            isFirstVisit = edges[childIdx].VisitCount == 0;
            AddVirtualLoss(_root, ref edges[childIdx]);
        }

        ref var childEdge = ref edges[childIdx];
        if (isFirstVisit)
        {
            _nodeCountPerThread[threadID]++;
            UpdateNodeStats(_root, ref childEdge, (double)childEdge.Value);
        }
        else
            UpdateNodeStats(_root, ref childEdge, VisitNode(threadID, ref state, _root.ChildNodes[childIdx], ref childEdge));
    }

    /// <summary>
    /// Recursively visits a node in the MCTS tree, handling expansion, selection, and backup.
    /// Implements the core MCTS algorithm with proper handling of pass moves and game termination.
    /// </summary>
    /// <param name="threadID">ID of the thread performing the visit</param>
    /// <param name="state">Current game state</param>
    /// <param name="node">Node to visit</param>
    /// <param name="edgeToNode">Edge leading to this node</param>
    /// <param name="afterPass">Whether this visit is after a pass move</param>
    /// <returns>Value to backup through the tree (from current player's perspective)</returns>
    double VisitNode(int threadID, ref State state, Node node, ref Edge edgeToNode, bool afterPass = false)
    {
        if (afterPass)
            state.Pass();
        else
            state.Update(ref edgeToNode.Move);

        var lockTaken = false;
        try
        {
            // Lock the node to prevent concurrent read/write access from other threads.
            // Use try-finally block to ensure the lock is released even if an exception occurs.
            Monitor.Enter(node, ref lockTaken);

            double value;
            Edge[] edges;
            if (node.Edges is null) // need to expand
            {
                edges = node.Expand(ref state.Position);
                if (edges[0].Move.Coord != BoardCoordinate.Pass)
                    SetPriorProbsAndValues(ref state, edges);
            }
            else
            {
                edges = node.Edges;
            }

            if (edges[0].Move.Coord == BoardCoordinate.Pass)  // pass
            {
                if (afterPass)  // gameover
                {
                    var outcome = GetOutcome(ref state);
                    edges[0].Label = EdgeLabel.Proved | (EdgeLabel)outcome;
                    edgeToNode.Label = EdgeLabel.Proved | (EdgeLabel)ToOpponentOutcome[outcome];

                    Monitor.Exit(node);
                    lockTaken = false;

                    value = OutcomeToReward[outcome];
                }
                else if (edges[0].IsProved)
                {
                    Monitor.Exit(node);
                    lockTaken = false;

                    value = OutcomeToReward[(int)(edges[0].Label ^ EdgeLabel.Proved)];
                }
                else
                {
                    Node childNode;
                    if (node.ChildNodes is null)
                    {
                        node.InitChildNodes();
                        childNode = node.CreateChildNode(0);
                    }
                    else
                        childNode = node.ChildNodes[0];

                    Monitor.Exit(node);
                    lockTaken = false;

                    value = VisitNode(threadID, ref state, childNode, ref edges[0], afterPass: true);

                    if (edges[0].IsProved)
                    {
                        var outcome = (int)(edges[0].Label ^ EdgeLabel.Proved);
                        edgeToNode.Label = EdgeLabel.Proved | (EdgeLabel)ToOpponentOutcome[outcome];
                    }
                }

                UpdatePassNodeStats(node, ref edges[0], value);
                return 1.0 - value;
            }

            // Handle non-pass moves
            var childIdx = SelectChildNode(node, ref edgeToNode.Label);
            ref var childEdge = ref edges[childIdx];
            var isFirstVisit = childEdge.VisitCount == 0;
            AddVirtualLoss(node, ref childEdge);

            if (isFirstVisit)
            {
                Monitor.Exit(node);
                lockTaken = false;

                _nodeCountPerThread[threadID]++;
                value = (double)childEdge.Value;
            }
            else if (childEdge.IsProved)
            {
                Monitor.Exit(node);
                lockTaken = false;

                value = OutcomeToReward[(int)(childEdge.Label ^ EdgeLabel.Proved)];
            }
            else
            {
                Node[] childNodes = node.ChildNodes is null ? node.InitChildNodes() : node.ChildNodes;
                var childNode = childNodes[childIdx] ?? node.CreateChildNode(childIdx);

                Monitor.Exit(node);
                lockTaken = false;

                value = VisitNode(threadID, ref state, childNode, ref childEdge);
            }

            UpdateNodeStats(node, ref childEdge, value);
            return 1.0 - value;
        }
        finally
        {
            if (lockTaken)
                Monitor.Exit(node);
        }
    }

    /// <summary>
    /// Sets prior probabilities and values for edges using the value function.
    /// Uses softmax to convert raw values into probabilities.
    /// </summary>
    /// <param name="state">Current game state</param>
    /// <param name="edges">Array of edges to set priors for</param>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    unsafe void SetPriorProbsAndValues(ref State state, Edge[] edges)
    {
        float value;
        float expValueSum = 0.0f;
        var expValues = stackalloc float[edges.Length];
        for (var i = 0; i < edges.Length; i++)
        {
            Debug.Assert(edges[i].Move.Coord != BoardCoordinate.Pass);

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
    /// Selects the best child node using the PUCT (PUCB) selection formula.
    /// Balances exploitation (Q-value) and exploration (prior probability and visit count).
    /// Also handles proved positions (win/loss/draw).
    /// </summary>
    /// <param name="parent">Parent node to select child from</param>
    /// <param name="parentEdgeLabel">Output parameter for parent's proof status</param>
    /// <returns>Index of the selected child</returns>
    static int SelectChildNode(Node parent, ref EdgeLabel parentEdgeLabel)
    {
        Debug.Assert(parent.Edges is not null);

        Edge[] edges = parent.Edges;
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
    /// Adds virtual loss to prevent multiple threads from selecting the same path.
    /// This improves parallel search efficiency by encouraging exploration diversity.
    /// </summary>
    /// <param name="parent">Parent node to add virtual loss to</param>
    /// <param name="childEdge">Child edge to add virtual loss to</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void AddVirtualLoss(Node parent, ref Edge childEdge)
    {
        Interlocked.Add(ref parent.VisitCount, VirtualLoss);
        Interlocked.Add(ref childEdge.VisitCount, VirtualLoss);
    }

    /// <summary>
    /// Updates node statistics after a simulation, removing virtual loss and adding the
    /// backup value to the edge's value sum.
    /// </summary>
    /// <param name="parent">Parent node to update</param>
    /// <param name="childEdge">Child edge to update</param>
    /// <param name="value">Value to back up through the edge</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void UpdateNodeStats(Node parent, ref Edge childEdge, double value)
    {
        if (VirtualLoss != 1)
        {
            Interlocked.Add(ref parent.VisitCount, unchecked(1 - VirtualLoss));
            Interlocked.Add(ref childEdge.VisitCount, unchecked(1 - VirtualLoss));
        }
        AtomicOperations.Add(ref childEdge.ValueSum, value);
    }

    /// <summary>
    /// Updates statistics for pass nodes (nodes where a pass move was made).
    /// </summary>
    /// <param name="parent">Parent node to update</param>
    /// <param name="childEdge">Child edge to update</param>
    /// <param name="reward">Reward value to add</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void UpdatePassNodeStats(Node parent, ref Edge childEdge, double reward)
    {
        Interlocked.Increment(ref parent.VisitCount);
        Interlocked.Increment(ref childEdge.VisitCount);
        AtomicOperations.Add(ref childEdge.ValueSum, reward);
    }

    /// <summary>
    /// Extracts the Principal Variation (PV) - the sequence of best moves from the search tree.
    /// </summary>
    /// <param name="node">Node to start extracting PV from</param>
    /// <param name="prevMove">Previous move to include in the PV</param>
    /// <returns>Sequence of moves representing the principal variation</returns>
    static IEnumerable<BoardCoordinate> GetPV(Node? node, BoardCoordinate prevMove = BoardCoordinate.Null)
    {
        if (prevMove != BoardCoordinate.Null)
            yield return prevMove;

        if (node is null || node.Edges is null)
            yield break;

        var childIdx = SelectBestChildNode(node);
        var childNode = node.ChildNodes?[childIdx];
        foreach (var move in GetPV(childNode, node.Edges[childIdx].Move.Coord))
            yield return move;
    }

    /// <summary>
    /// Selects the best child node for move selection (not exploration).
    /// Prioritizes proven wins, avoids proven losses, and selects based on visit count and reward.
    /// </summary>
    /// <param name="parent">Parent node to select best child from</param>
    /// <returns>Index of the best child node</returns>
    static int SelectBestChildNode(Node parent)
    {
        Debug.Assert(parent.Edges is not null);

        Edge[] edges = parent.Edges;
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
    /// Converts an EdgeLabel to the corresponding GameResult.
    /// </summary>
    /// <param name="label">Edge label to convert</param>
    /// <returns>Corresponding game result</returns>
    /// <exception cref="NotImplementedException">Thrown for unknown edge labels</exception>
    static GameResult EdgeLabelToGameResult(EdgeLabel label)
    {
        return label switch
        {
            EdgeLabel.NotProved => GameResult.NotOver,
            EdgeLabel.Win => GameResult.Win,
            EdgeLabel.Loss => GameResult.Loss,
            EdgeLabel.Draw => GameResult.Draw,
            _ => throw new NotImplementedException()
        };
    }

    /// <summary>
    /// Determines the game outcome from a terminal position.
    /// </summary>
    /// <param name="state">Terminal game state</param>
    /// <returns>Outcome flag (OutcomeWin, OutcomeLoss, or OutcomeDraw)</returns>
    static int GetOutcome(ref State state)
    {
        Debug.Assert(state.Position.IsGameOver);

        var score = state.Position.DiscDiff;
        if (score == 0)
            return OutcomeDraw;
        return score > 0 ? OutcomeWin : OutcomeLoss;
    }
}