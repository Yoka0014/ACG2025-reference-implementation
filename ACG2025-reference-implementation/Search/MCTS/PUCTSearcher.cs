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

internal static class PUCTConstants
{
    public const float PUCTFactor = 1.0f;
    public const uint VirtualLoss = 1;

    public const int OutcomeWin = 1;
    public const int OutcomeLoss = 1 << 1;
    public const int OutcomeDraw = 1 << 2;
    public static ReadOnlySpan<float> OutcomeToReward => [float.NaN, 1.0f, 0.0f, float.NaN, 0.5f];
    public static ReadOnlySpan<int> ToOpponentOutcome => [0, OutcomeLoss, OutcomeWin, 0, OutcomeDraw];
}

internal class MoveEval(IEnumerable<BoardCoordinate> pv) : IComparable<MoveEval>
{
    public BoardCoordinate Move { get; init; }
    public double Effort { get; init; }
    public long SimulationCount { get; init; }
    public double ExpectedReward { get; init; }
    public GameResult GameResult { get; init; }
    public ReadOnlySpan<BoardCoordinate> PV => _pv;

    readonly BoardCoordinate[] _pv = [.. pv];

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

internal class SearchResult(MoveEval rootEval, IEnumerable<MoveEval> childEvals)
{
    public MoveEval RootEval { get; } = rootEval;
    public ReadOnlySpan<MoveEval> ChildEvals => _childEvals;
    readonly MoveEval[] _childEvals = [.. childEvals];
}

internal class PUCTSearcher
{
    const float Epsilon = 1.0e-6f;

    public event EventHandler<SearchResult?> SearchResultUpdated = delegate { };

    public int SearchResultUpdateIntervalCs { get; set; }

    public double RootDirchletAlpha { get; set; } = 0.3;
    public double RootExplorationFraction { get; set; } = 0.25;

    ValueFunction _valueFunc;

    Node? _root;
    EdgeLabel _rootEdgeLabel;
    Position _rootPos;

    volatile bool _isSearching;
    CancellationTokenSource? _cts;

    int _searchStartTimeMs = 0;
    int _searchEndTimeMs = 0;

    int _numThreads = Environment.ProcessorCount;
    long[] _nodeCountPerThread;
    long _maxSimulationCount;
    long _simulationCount;

    public PUCTSearcher(ValueFunction valueFunc)
    {
        _valueFunc = valueFunc;
        _nodeCountPerThread = new long[_numThreads];
    }

    public bool IsSearching => _isSearching;

    public int SearchElapsedMs => _isSearching ? Environment.TickCount - _searchStartTimeMs : _searchEndTimeMs - _searchStartTimeMs;

    public long NodeCount => _nodeCountPerThread.Sum();

    public double Nps => NodeCount / (SearchElapsedMs * 1.0e-3);

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

    public void SendStopSearchSignal() => _cts?.Cancel();

    public void SetRootPosition(ref Position pos)
    {
        _rootPos = pos;
        _root = new Node();
        _rootEdgeLabel = EdgeLabel.NotProved;
        InitRootChildNodes();
        Array.Clear(_nodeCountPerThread);
    }

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

    public SearchResult? GetSearchResult()
    {
        if (_root is null || _root.Edges is null)
            return null;

        var rootEdgeLabel = _rootEdgeLabel;

        if ((rootEdgeLabel & EdgeLabel.Proved) != 0)
        {
            // rootEdgeLabelには，敵視点の勝敗が記録されるため，ここで手番視点に変換が必要．
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

    public Move? SelectBestMove() => _root?.Edges?[SelectBestChildNode(_root)].Move;

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

    public void Search(long numSimulations, int timeLimitCs)
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

    public void SearchOnSingleThread(uint numPlayouts)
    {
        var rootState = new State(_rootPos, _valueFunc.NTupleManager);
        var state = new State(_valueFunc.NTupleManager);
        for (var i = 0u; i < numPlayouts && _rootEdgeLabel == EdgeLabel.NotProved; i++)
        {
            rootState.CopyTo(ref state);
            VisitRootNode(0, ref state);
            _simulationCount++;
        }
    }

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

    bool CanStop(int timeLimitMs)
    {
        Debug.Assert(_cts is not null);

        return _cts.IsCancellationRequested
            || (_rootEdgeLabel & EdgeLabel.Proved) != 0
            || SearchElapsedMs >= timeLimitMs
            || _simulationCount >= _maxSimulationCount;
    }

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

        // 探索性を向上させるため，ルート直下の事前確率にディリクレノイズを加える．
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

    double VisitNode(int threadID, ref State state, Node node, ref Edge edgeToNode, bool afterPass = false)
    {
        if (afterPass)
            state.Pass();
        else
            state.Update(ref edgeToNode.Move);

        var lockTaken = false;
        try
        {
            // 他スレッドから同時に読み書きが行われないようにノードをロック．
            // 例外が発生してもロックが外れることを保証するために，クリティカルセクション全体をtry~finallyブロックで囲う必要あり．
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

            // not pass
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

        // softmax
        for (var i = 0; i < edges.Length; i++)
            edges[i].PriorProb = (Half)(expValues[i] / expValueSum);
    }

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
                // if there is a win edge from player view, it determines loss from opponent. 
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

            // calculate PUCB score.
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void AddVirtualLoss(Node parent, ref Edge childEdge)
    {
        Interlocked.Add(ref parent.VisitCount, VirtualLoss);
        Interlocked.Add(ref childEdge.VisitCount, VirtualLoss);
    }

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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void UpdatePassNodeStats(Node parent, ref Edge childEdge, double reward)
    {
        Interlocked.Increment(ref parent.VisitCount);
        Interlocked.Increment(ref childEdge.VisitCount);
        AtomicOperations.Add(ref childEdge.ValueSum, reward);
    }

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

    static int GetOutcome(ref State state)
    {
        Debug.Assert(state.Position.IsGameOver);

        var score = state.Position.DiscDiff;
        if (score == 0)
            return OutcomeDraw;
        return score > 0 ? OutcomeWin : OutcomeLoss;
    }
}