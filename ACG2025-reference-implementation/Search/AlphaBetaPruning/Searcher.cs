namespace ACG2025_reference_implementation.Search.AlphaBetaPruning;

using System;
using System.Text;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Threading.Tasks;
using System.Runtime.CompilerServices;

using ACG2025_reference_implementation.Reversi;
using ACG2025_reference_implementation.Evaluation;

/// <summary>
/// Represents the result of an alpha-beta search, containing the best move, evaluation value,
/// search depth, node count, and principal variation.
/// </summary>
/// <param name="pv">The principal variation (sequence of best moves)</param>
internal class SearchResult(IEnumerable<BoardCoordinate> pv)
{
    /// <summary>
    /// Gets the best move found during the search.
    /// </summary>
    public BoardCoordinate BestMove { get; init; }

    /// <summary>
    /// Gets the evaluation value of the position after the best move.
    /// </summary>
    public int SearchValue { get; init; }

    /// <summary>
    /// Gets the depth to which the search was conducted.
    /// </summary>
    public int Depth { get; init; }

    /// <summary>
    /// Gets the total number of nodes examined during the search.
    /// </summary>
    public long NodeCount { get; init; }

    /// <summary>
    /// Gets the elapsed time in milliseconds for the search.
    /// </summary>
    public int ElapsedMs { get; init; }

    /// <summary>
    /// Gets the principal variation (sequence of best moves) as a read-only collection.
    /// </summary>
    public ReadOnlyCollection<BoardCoordinate> PV => new(_pv);

    readonly List<BoardCoordinate> _pv = [.. pv];
};

/// <summary>
/// Represents a Principal Variation (PV) - a sequence of moves considered optimal by the search.
/// Used internally during alpha-beta search to track and build the best line of play.
/// </summary>
struct PV
{
    /// <summary>
    /// Gets the number of moves currently stored in the principal variation.
    /// </summary>
    public int Count { get; private set; }

    /// <summary>
    /// Gets or sets a value indicating whether the PV was cut short by a transposition table hit.
    /// When true, the remaining PV can potentially be reconstructed from the transposition table.
    /// </summary>
    public bool CutByTT { get; set; }

    Moves _moves;

    /// <summary>
    /// Gets the move at the specified index in the principal variation.
    /// </summary>
    /// <param name="idx">The zero-based index of the move to retrieve</param>
    /// <returns>The board coordinate of the move at the specified index</returns>
    /// <exception cref="IndexOutOfRangeException">Thrown when idx is less than 0 or greater than or equal to Count</exception>
    public readonly BoardCoordinate this[int idx]
    {
        get
        {
            if (idx < 0 || idx >= Count)
                throw new IndexOutOfRangeException();

            return _moves[idx];
        }
    }

    /// <summary>
    /// Clears the principal variation, removing all moves and resetting the CutByTT flag.
    /// </summary>
    public void Clear()
    {
        Count = 0;
        CutByTT = false;
    }

    /// <summary>
    /// Copies the contents of this principal variation to the destination PV.
    /// </summary>
    /// <param name="dest">The destination PV to copy to</param>
    public readonly void CopyTo(ref PV dest)
    {
        dest.Count = Count;
        dest.CutByTT = CutByTT;

        for (var i = 0; i < Count; i++)
            dest._moves[i] = _moves[i];
    }

    /// <summary>
    /// Determines whether the principal variation contains the specified move.
    /// </summary>
    /// <param name="move">The move to search for</param>
    /// <returns>true if the move is found in the PV; otherwise, false</returns>
    public readonly bool Contains(BoardCoordinate move)
    {
        for (var i = 0; i < Count; i++)
        {
            if (_moves[i] == move)
                return true;
        }
        return false;
    }

    /// <summary>
    /// Adds a move to the end of the principal variation.
    /// </summary>
    /// <param name="move">The move to add to the PV</param>
    public void Add(BoardCoordinate move) => _moves[Count++] = move;

    /// <summary>
    /// Adds all moves from another principal variation to this one.
    /// Also copies the CutByTT flag from the source PV.
    /// </summary>
    /// <param name="pv">The source PV to copy moves from</param>
    public void AddRange(ref PV pv)
    {
        for (var i = 0; i < pv.Count; i++)
            Add(pv._moves[i]);
        CutByTT = pv.CutByTT;
    }

    /// <summary>
    /// Removes all moves from the specified index to the end of the principal variation.
    /// Also resets the CutByTT flag to false.
    /// </summary>
    /// <param name="start">The starting index from which to remove moves</param>
    public void RemoveUnder(int start)
    {
        Debug.Assert(0 <= start && start < Count);

        Count -= Count - start;
        CutByTT = false;
    }

    /// <summary>
    /// Updates the given position by playing all moves in the principal variation.
    /// Handles pass moves automatically when the current player has no legal moves.
    /// </summary>
    /// <param name="pos">The position to update by playing the PV moves</param>
    public readonly void UpdatePositionAlongPV(ref Position pos)
    {
        for (int i = 0; i < Count; i++)
        {
            if (pos.CanPass)
                pos.Pass();

            var move = new Move(_moves[i]);
            pos.CalcFlip(ref move);
            pos.Update(ref move);
        }
    }

    /// <summary>
    /// Returns a string representation of the principal variation using 2D coordinate notation.
    /// </summary>
    /// <returns>A string containing all moves in the PV using board coordinate notation</returns>
    public override readonly string ToString()
    {
        var sb = new StringBuilder();
        for (var i = 0; i < Count; i++)
            sb.Append(Utils.Coordinate1DTo2D(_moves[i]));
        return sb.ToString();
    }

    /// <summary>
    /// Inline array structure for storing board coordinates efficiently.
    /// Uses unsafe code for high-performance access during search operations.
    /// </summary>
    [InlineArray(Constants.NumInitialCells)]
    struct Moves
    {
        BoardCoordinate _coord;

#pragma warning disable CS9181 // Suppress warning for using custom indexer for performance optimization
        /// <summary>
        /// Gets or sets the move at the specified index using unsafe pointer arithmetic for performance.
        /// </summary>
        /// <param name="idx">The index of the move to access</param>
        /// <returns>The board coordinate at the specified index</returns>
        /// <exception cref="IndexOutOfRangeException">Thrown in debug builds when the index is out of range</exception>
        public unsafe BoardCoordinate this[int idx]
        {
            get
            {
#if DEBUG
                if (idx < 0 || idx >= Constants.NumInitialCells)
                    throw new IndexOutOfRangeException();
#endif

                fixed (Moves* self = &this)
                {
                    var moves = (BoardCoordinate*)self;
                    return moves[idx];
                }
            }

            set
            {
#if DEBUG
                if (idx < 0 || idx >= Constants.NumInitialCells)
                    throw new IndexOutOfRangeException();
#endif

                fixed (Moves* self = &this)
                {
                    var moves = (BoardCoordinate*)self;
                    moves[idx] = value;
                }
            }
        }
#pragma warning restore CS9181 
    }
}

/// <summary>
/// Alpha-beta search engine for Reversi using negamax framework with various optimizations.
/// Features include transposition table, iterative deepening, move ordering, and separate
/// handling for midgame and endgame phases.
/// </summary>
/// <param name="valueFunc">The evaluation function to use for position assessment</param>
/// <param name="ttSizeBytes">The size of the transposition table in bytes</param>
internal class Searcher(ValueFunction valueFunc, long ttSizeBytes)
{
    /// <summary>
    /// Represents positive infinity in the search value range.
    /// </summary>
    public const int ValueInf = short.MaxValue;
    
    /// <summary>
    /// Represents an invalid search value, used for error conditions.
    /// </summary>
    public const int ValueInvalid = int.MaxValue;
    
    /// <summary>
    /// The minimum search depth for iterative deepening.
    /// </summary>
    public const int DepthMin = 4;
    
    /// <summary>
    /// The threshold for switching from midgame to endgame search algorithms.
    /// When empty cell count is below this value, endgame-specific optimizations are used.
    /// </summary>
    public const int EndgameDepth = 15;

    /// <summary>
    /// In the midgame, switch to shallow search when remaining depth is at or below this value.
    /// </summary>
    const int MidgameShallowDepth = 4;

    /// <summary>
    /// In the endgame, switch to shallow search when remaining depth is at or below this value.
    /// </summary>
    const int EndgameShallowDepth = 7;

    /// <summary>
    /// Do not perform iterative deepening when moves to game end are at or below this value.
    /// </summary>
    const int MateSearchDepth = 10;

    /// <summary>
    /// Do not perform move ordering when remaining search depth is at or below this value.
    /// </summary>
    const int NoMoveOrderingDepth = 2;

    /// <summary>
    /// Whether to use the position evaluation function for move ordering.
    /// </summary>
    const bool UseEvaluatorForMoveOrdering = true;

    /// <summary>
    /// Check the search interruption flag at nodes with this remaining search depth.
    /// </summary>
    const int SuspendFlagCheckDepth = 5;

    /// <summary>
    /// Gets or sets whether to aggressively save best moves in the transposition table.
    /// When true, best moves are recorded even at nodes close to leaves, making PV reconstruction easier.
    /// However, this doesn't affect playing strength and may cause time loss due to additional TT accesses.
    /// </summary>
    public bool AggressivePVSave { get; set; } = true;

    /// <summary>
    /// Gets or sets the interval in milliseconds for PV update notifications during search.
    /// </summary>
    public int PVNotificationIntervalMs { get; set; } = 100;

    /// <summary>
    /// Event raised when search results are updated during iterative deepening.
    /// </summary>
    public event EventHandler<SearchResult> SearchResultUpdated = delegate { };

    /// <summary>
    /// Gets the total number of nodes examined in the current or last search.
    /// </summary>
    public long NodeCount => _nodeCount;
    
    /// <summary>
    /// Gets a value indicating whether a search operation is currently in progress.
    /// </summary>
    public bool IsSearching => _isSearching;

    readonly ValueFunction _valueFunc = valueFunc;
    TranspositionTable _tt = new(ttSizeBytes);
    Position _rootPos;
    bool _rootWasSet = false;
    long _nodeCount = 0;
    volatile bool _isSearching = false;
    volatile bool _suspendFlag = false;

    /// <summary>
    /// Attempts to resize the transposition table to the specified size.
    /// The resize operation will fail if a search is currently in progress.
    /// </summary>
    /// <param name="ttSizeBytes">The new size of the transposition table in bytes</param>
    /// <returns>true if the resize was successful; false if a search is in progress</returns>
    public bool TryResizeTranspositionTable(long ttSizeBytes)
    {
        if (IsSearching)
            return false;
        _tt = new TranspositionTable(ttSizeBytes);
        return true;
    }

    /// <summary>
    /// Requests the search to stop at the next convenient point.
    /// The search will check this flag periodically and terminate gracefully.
    /// </summary>
    public void Stop() => _suspendFlag = true;

    /// <summary>
    /// Sets the root position for the search. If a root position was previously set,
    /// the transposition table is cleared to avoid hash collisions from different games.
    /// </summary>
    /// <param name="pos">The position to set as the search root</param>
    public void SetRoot(ref Position pos)
    {
        if (_rootWasSet)
            _tt.Clear();
        _rootPos = pos;
        _rootWasSet = true;
    }

    /// <summary>
    /// Attempts to update the root position by making the specified move.
    /// </summary>
    /// <param name="moveCoord">The coordinate of the move to make</param>
    /// <returns>true if the move was legal and the root was updated; false if the move was illegal</returns>
    /// <exception cref="InvalidOperationException">Thrown if no root position has been set</exception>
    public bool TryUpdateRoot(BoardCoordinate moveCoord)
    {
        if (!_rootWasSet)
            throw new InvalidOperationException("Root position has not been set.");

        if (!_rootPos.IsLegalMoveAt(moveCoord))
            return false;

        var move = new Move(moveCoord);
        _rootPos.CalcFlip(ref move);
        UpdateRoot(ref move);
        return true;
    }

    /// <summary>
    /// Updates the root position by applying the specified move.
    /// Also increments the transposition table generation to invalidate old entries.
    /// </summary>
    /// <param name="move">The move to apply to the root position</param>
    /// <exception cref="InvalidOperationException">Thrown if no root position has been set</exception>
    public void UpdateRoot(ref Move move)
    {
        if (!_rootWasSet)
            throw new InvalidOperationException("Root position has not been set.");

        if (move.Coord == BoardCoordinate.Pass)
            _rootPos.Pass();
        else
            _rootPos.Update(ref move);

        _tt.IncGen();
    }

    /// <summary>
    /// Updates the root position by passing the turn (no move made).
    /// </summary>
    public void PassRoot()
    {
        var move = new Move(BoardCoordinate.Pass);
        UpdateRoot(ref move);
    }

    /// <summary>
    /// Performs an asynchronous search to the specified depth.
    /// The search runs on a background thread and calls the completion handler when finished.
    /// </summary>
    /// <param name="maxDepth">The maximum depth to search</param>
    /// <param name="onCompleted">The event handler to call when the search completes</param>
    /// <returns>A task that represents the search operation, containing the search result when completed</returns>
    /// <exception cref="InvalidOperationException">Thrown if a search is already in progress</exception>
    public async Task<SearchResult?> SearchAsync(int maxDepth, EventHandler<SearchResult> onCompleted)
    {
        if (IsSearching)
            throw new InvalidOperationException("Search is already in progress.");

        _suspendFlag = false;
        _isSearching = true;

        SearchResult? res = null;
        await Task.Run(() =>
        {
            res = Search(maxDepth);
            onCompleted(this, res);
        }).ConfigureAwait(false);

        return res;
    }

    /// <summary>
    /// Performs a synchronous alpha-beta search to the specified depth using iterative deepening.
    /// The search alternates between midgame and endgame algorithms based on the number of empty cells.
    /// </summary>
    /// <param name="maxDepth">The maximum depth to search</param>
    /// <returns>A SearchResult containing the best move, evaluation, and search statistics</returns>
    /// <exception cref="InvalidOperationException">Thrown if no root position has been set</exception>
    public SearchResult Search(int maxDepth)
    {
        if (!_rootWasSet)
            throw new InvalidOperationException("Root position has not been set.");

        _suspendFlag = false;
        _isSearching = true;
        _nodeCount = 0;

        SearchResult res;
        var pv = new PV();
        int depth;
        var rootPos = _rootPos;
        var state = new State(_valueFunc.NTupleManager);
        state.Init(ref rootPos);

        var startTimeMs = Environment.TickCount;
        int elapsedMs = 0;
        var lastPVNotificationTimeMs = Environment.TickCount;
        var stopFlag = false;

        maxDepth = Math.Min(rootPos.EmptyCellCount, maxDepth);

        if (maxDepth <= DepthMin)
        {
            if (state.Position.EmptyCellCount > EndgameDepth)
                SearchWithTT<Midgame, NotAfterPass>(ref state, -ValueInf, ValueInf, ref pv, maxDepth, ref _nodeCount, ref stopFlag);
            else
                SearchWithTT<Endgame, NotAfterPass>(ref state, -ValueInf, ValueInf, ref pv, maxDepth, ref _nodeCount, ref stopFlag);

            ref var entry = ref _tt.GetEntry(ref state.Position, out var hit);

            Debug.Assert(hit);    // Root position value should always hit since it's recorded last in the transposition table.
            Debug.Assert(entry.HasBestMove);   // Root position TT entry should always contain the best move.

            res = new SearchResult(CreatePVList(ref pv, maxDepth))
            {
                BestMove = entry.Move,
                SearchValue = entry.Lower,
                Depth = entry.Depth,
                NodeCount = _nodeCount,
                ElapsedMs = Environment.TickCount - startTimeMs
            };

            _isSearching = false;

            return res;
        }

        var idMax = (maxDepth == state.Position.EmptyCellCount) ? Math.Max(0, state.Position.EmptyCellCount - MateSearchDepth) : maxDepth;
        depth = (idMax % 2 == 0) ? DepthMin : DepthMin + 1;

        var bestMove = BoardCoordinate.Null;
        int value = ValueInvalid;
        for(; depth <= idMax; depth += 2)
        {
            pv.Clear();
            if(state.Position.EmptyCellCount > EndgameDepth)
                SearchWithTT<Midgame, NotAfterPass>(ref state, -ValueInf, ValueInf, ref pv, depth, ref _nodeCount, ref stopFlag);
            else
                SearchWithTT<Endgame, NotAfterPass>(ref state, -ValueInf, ValueInf, ref pv, depth, ref _nodeCount, ref stopFlag);

            if(stopFlag)
                break;

            ref var entry = ref _tt.GetEntry(ref state.Position, out var hit);

            Debug.Assert(hit);   
            Debug.Assert(entry.HasBestMove);   

            bestMove = entry.Move;
            value = entry.Lower;

            var timeNowMs = Environment.TickCount;
            elapsedMs = timeNowMs - startTimeMs;

            if(timeNowMs - lastPVNotificationTimeMs >= PVNotificationIntervalMs)
            {
                res = new SearchResult(CreatePVList(ref pv, depth))
                {
                    BestMove = bestMove,
                    SearchValue = value,
                    Depth = depth,
                    NodeCount = _nodeCount,
                    ElapsedMs = elapsedMs
                };
                SearchResultUpdated(this, res);
                lastPVNotificationTimeMs = timeNowMs;
            }
        }

        depth -= 2;

        if(!stopFlag && depth < maxDepth)
        {
            depth = maxDepth;

            if(state.Position.EmptyCellCount > EndgameDepth)
                SearchWithTT<Midgame, NotAfterPass>(ref state, -ValueInf, ValueInf, ref pv, depth, ref _nodeCount, ref stopFlag);
            else
                SearchWithTT<Endgame, NotAfterPass>(ref state, -ValueInf, ValueInf, ref pv, depth, ref _nodeCount, ref stopFlag);

            if (!stopFlag)
            {
                ref TTEntry entry = ref _tt.GetEntry(ref state.Position, out var hit);

                Debug.Assert(hit);
                Debug.Assert(entry.HasBestMove);

                bestMove = entry.Move;
                value = entry.Lower;

                var timeNowMs = Environment.TickCount;
                elapsedMs = timeNowMs - startTimeMs;
            }
        }

        res = new SearchResult(CreatePVList(ref pv, maxDepth))
        {
            BestMove = bestMove,
            SearchValue = value,
            Depth = depth,
            NodeCount = _nodeCount,
            ElapsedMs = elapsedMs
        };

        _isSearching = false;

        return res;
    }

    /// <summary>
    /// Evaluates the given game state using the position evaluation function.
    /// </summary>
    /// <param name="state">The game state to evaluate</param>
    /// <returns>The evaluation value of the position</returns>
    int Evaluate(ref State state) => _valueFunc.F(state.FeatureVector);

    /// <summary>
    /// Evaluates and scores a list of moves for move ordering purposes.
    /// Uses position evaluation and mobility heuristics to rank moves.
    /// </summary>
    /// <typeparam name="Phase">The game phase (Midgame or Endgame)</typeparam>
    /// <param name="state">The current game state</param>
    /// <param name="moves">The moves to evaluate</param>
    /// <param name="scores">Output array to store the evaluation scores</param>
    /// <param name="depth">The remaining search depth</param>
    void EvaluateMoves<Phase>(ref State state, Span<Move> moves, Span<int> scores, int depth)
        where Phase : struct, IGamePhase
    {
        // Evaluate each move.
        // Currently uses the position evaluation function value and number of possible moves for evaluation.
        for (int i = 0; i < moves.Length; i++)
        {
            ref Move move = ref moves[i];
            scores[i] = 0;

            Debug.Assert(move.Flip != 0UL);
            Debug.Assert(state.Position.IsLegalMoveAt(move.Coord));

            state.Position.Update(ref move);

            if (typeof(Phase) == typeof(Midgame))
            {
                if (UseEvaluatorForMoveOrdering && depth > MidgameShallowDepth)
                {
                    state.FeatureVector.Update(ref move);
                    scores[i] += -_valueFunc.F(state.FeatureVector);
                    state.FeatureVector.Undo(ref move);
                }
            }

            scores[i] += Constants.MaxLegalMoves - state.Position.GetNumNextMoves();

            state.Position.Undo(ref move);
        }
    }

    /// <summary>
    /// Performs the next level of search, automatically choosing between transposition table search
    /// and shallow search based on remaining depth and game phase.
    /// </summary>
    /// <typeparam name="Phase">The game phase (Midgame or Endgame)</typeparam>
    /// <param name="state">The current game state</param>
    /// <param name="alpha">The alpha bound for alpha-beta pruning</param>
    /// <param name="beta">The beta bound for alpha-beta pruning</param>
    /// <param name="pv">The principal variation being built</param>
    /// <param name="depth">The remaining search depth</param>
    /// <param name="nodeCount">Reference to the node count for statistics</param>
    /// <param name="stopFlag">Reference to the stop flag for search interruption</param>
    /// <returns>The evaluation value of the position</returns>
    int SearchNext<Phase>(ref State state, int alpha, int beta, ref PV pv, int depth, ref long nodeCount, ref bool stopFlag)
        where Phase : struct, IGamePhase
    {
        Debug.Assert(alpha < beta);

        nodeCount++;

        if (typeof(Phase) == typeof(Midgame))
        {
            if (depth > MidgameShallowDepth)
            {
                if (state.Position.EmptyCellCount > EndgameDepth)
                    return SearchWithTT<Phase, NotAfterPass>(ref state, alpha, beta, ref pv, depth, ref nodeCount, ref stopFlag);
                else
                    return SearchWithTT<Endgame, NotAfterPass>(ref state, alpha, beta, ref pv, depth, ref nodeCount, ref stopFlag);
            }
            else
            {
                return SearchShallow<Phase, NotAfterPass>(ref state, alpha, beta, ref pv, depth, ref nodeCount, ref stopFlag);
            }
        }
        else
        {
            if (depth > EndgameShallowDepth)
                return SearchWithTT<Endgame, NotAfterPass>(ref state, alpha, beta, ref pv, depth, ref nodeCount, ref stopFlag);
            else
                return SearchShallow<Endgame, NotAfterPass>(ref state, alpha, beta, ref pv, depth, ref nodeCount, ref stopFlag);
        }
    }

    int SearchShallowNext<Phase>(ref State state, int alpha, int beta, ref PV pv, int depth, ref long nodeCount, ref bool stopFlag)
        where Phase : struct, IGamePhase
    {
        if (typeof(Phase) == typeof(Midgame) && state.Position.EmptyCellCount > EndgameDepth)
            return SearchShallow<Phase, NotAfterPass>(ref state, alpha, beta, ref pv, depth, ref nodeCount, ref stopFlag);
        else
            return SearchShallow<Endgame, NotAfterPass>(ref state, alpha, beta, ref pv, depth, ref nodeCount, ref stopFlag);
    }

    /// <summary>
    /// Performs alpha-beta search with transposition table lookup and storage.
    /// This is the main search routine that handles TT probing, move ordering, and recursive search with pruning.
    /// </summary>
    /// <typeparam name="Phase">The game phase (Midgame or Endgame)</typeparam>
    /// <typeparam name="PassFlag">Indicates if this search follows a pass move</typeparam>
    /// <param name="state">The current game state</param>
    /// <param name="alpha">The alpha bound for alpha-beta pruning</param>
    /// <param name="beta">The beta bound for alpha-beta pruning</param>
    /// <param name="pv">The principal variation being built</param>
    /// <param name="depth">The remaining search depth</param>
    /// <param name="nodeCount">Reference to the node count for statistics</param>
    /// <param name="stopFlag">Reference to the stop flag for search interruption</param>
    /// <returns>The evaluation value of the position</returns>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    int SearchWithTT<Phase, PassFlag>(ref State state, int alpha, int beta, ref PV pv, int depth, ref long nodeCount, ref bool stopFlag)
        where Phase : struct, IGamePhase where PassFlag : struct, IPassFlag
    {
        Debug.Assert(depth >= 0);

        if (depth == SuspendFlagCheckDepth)
        {
            stopFlag = _suspendFlag;
            if (stopFlag)
                return ValueInvalid;
        }

        ref TTEntry entry = ref _tt.GetEntry(ref state.Position, out var hit);
        if (hit && entry.Generation == _tt.Generation && entry.Depth == depth)
        {
            // For previously searched positions, get the evaluation range from the transposition table entry.
            int lower = entry.Lower;
            int upper = entry.Upper;

            // If the true evaluation value of state.pos_ is v, then lower <= v <= upper holds.
            // Therefore, if upper <= alpha we can alpha cut, if beta <= lower we can beta cut.
            // Also, when lower == upper, v == lower == upper, so we can return lower (or upper) directly.

            if (upper <= alpha)
            {
                pv.CutByTT = true;
                return upper;
            }

            if (beta <= lower || lower == upper)
            {
                pv.CutByTT = true;
                return lower;
            }

            // Narrow the search window.
            alpha = Math.Max(alpha, lower);
            beta = Math.Min(beta, upper);
        }

        Span<Move> moves = stackalloc Move[Constants.MaxLegalMoves];
        moves = moves[..state.Position.GetNextMoves(ref moves)];
        Span<int> moveScores = stackalloc int[moves.Length];

        if (moves.Length == 0)  // pass
        {
            if (typeof(PassFlag) == typeof(AfterPass))   // Two consecutive passes -> game end
                return DiscDiffToValue(state.Position.DiscDiff);

            state.Pass();

            int pass_value = -SearchWithTT<Phase, AfterPass>(ref state, -beta, -alpha, ref pv, depth, ref nodeCount, ref stopFlag);

            state.Pass();

            return pass_value;
        }

        if (depth == 0)  // Leaf node
            return Evaluate(ref state);

        // From here on, processing for internal nodes.

        int ply = pv.Count;

        // Calculate flipped discs.
        InitMoves(ref state.Position, moves);

        if (hit && entry.HasBestMove)
        {
            // If the best move from the previous search is recorded in the transposition table, place it at the front.
            PlaceToHead(moves, entry.Move);
        }
        else
        {
            // Score all moves.
            EvaluateMoves<Phase>(ref state, moves, moveScores, depth);
            SortMoves(moves, moveScores);
        }

        // Assume the first move in the move list is the best move.
        ref var bestMove = ref moves[0];
        int maxValue, value, a = alpha;
        long childNodeCount = 0;

        state.Update(ref bestMove);
        pv.Add(bestMove.Coord);

        // Search the best move candidate with the current search window.
        maxValue = value = -SearchNext<Phase>(ref state, -beta, -a, ref pv, depth - 1, ref childNodeCount, ref stopFlag);

        state.Undo(ref bestMove);

        if (stopFlag)
            return ValueInvalid;

        // beta cut
        if (value >= beta)
        {
            // Since beta cut occurred, record that this position's evaluation value is at least value or higher.
            // If value == ValueInf, it's a guaranteed win, so also record best_move.
            if (value != ValueInf)
                _tt.SaveAt(ref entry, ref state.Position, value, ValueInf, depth, childNodeCount);
            else
                _tt.SaveAt(ref entry, ref state.Position, bestMove.Coord, ValueInf, ValueInf, depth, childNodeCount);

            nodeCount += childNodeCount;
            return value;
        }

        // This position's evaluation value is at least value or higher, so update the alpha value.
        if (value > a)
            a = value;

        // If the best move from the previous search was recorded in the transposition table but beta cut didn't occur,
        // perform scoring for the remaining moves.
        var offset = 0;
        if (hit && entry.HasBestMove && moves.Length > 1)
        {
            EvaluateMoves<Phase>(ref state, moves[1..], moveScores[1..], depth);
            SortMoves(moves[1..], moveScores[1..]);
            offset = 1;
        }

        // Search the remaining moves.
        var currentPV = new PV();
        for (var i = offset; i < moves.Length; i++)
        {
            ref Move move = ref moves[i];
            state.Update(ref move);

            // Check with NWS (Null Window Search) whether it exceeds the current alpha value.
            currentPV.Clear();
            currentPV.Add(move.Coord);
            value = -SearchNext<Phase>(ref state, -a - 1, -a, ref currentPV, depth - 1, ref childNodeCount, ref stopFlag);

            if (stopFlag)
                return ValueInvalid;

            // beta cut
            if (value >= beta)
            {
                state.Undo(ref move);

                if (value != ValueInf)
                    _tt.SaveAt(ref entry, ref state.Position, value, ValueInf, depth, childNodeCount);
                else
                    _tt.SaveAt(ref entry, ref state.Position, move.Coord, ValueInf, ValueInf, depth, childNodeCount);

                pv.RemoveUnder(ply);
                pv.AddRange(ref currentPV);
                nodeCount += childNodeCount;
                return value;
            }

            // Since it was found to exceed the current alpha value, re-search with the current search window.
            if (value > a)
            {
                a = value;

                currentPV.Clear();
                currentPV.Add(move.Coord);
                value = -SearchNext<Phase>(ref state, -beta, -a, ref currentPV, depth - 1, ref childNodeCount, ref stopFlag);

                if (stopFlag)
                    return ValueInvalid;

                // beta cut
                if (value >= beta)
                {
                    state.Undo(ref move);

                    if (value != ValueInf)
                        _tt.SaveAt(ref entry, ref state.Position, value, ValueInf, depth, childNodeCount);
                    else
                        _tt.SaveAt(ref entry, ref state.Position, move.Coord, ValueInf, ValueInf, depth, childNodeCount);

                    pv.RemoveUnder(ply);
                    pv.AddRange(ref currentPV);
                    nodeCount += childNodeCount;
                    return value;
                }
            }

            // Update the best move
            if (value > maxValue)
            {
                bestMove = move;
                maxValue = value;
                a = Math.Max(a, value);

                pv.RemoveUnder(ply);
                pv.AddRange(ref currentPV);
            }

            state.Undo(ref move);
        }

        if (maxValue > alpha)   // Evaluation value determined within search window -> true evaluation value. Don't use max_value >= alpha (would include cases where alpha was returned due to beta cut in descendant nodes)
            _tt.SaveAt(ref entry, ref state.Position, bestMove.Coord, maxValue, maxValue, depth, childNodeCount);
        else if (maxValue != -ValueInf)    // Since it fell below alpha, it's determined that this position's evaluation value is at most max_value or lower.
            _tt.SaveAt(ref entry, ref state.Position, -ValueInf, maxValue, depth, childNodeCount);
        else    // If max_value is -ValueInf, then this position is determined to be a guaranteed loss.
            _tt.SaveAt(ref entry, ref state.Position, bestMove.Coord, -ValueInf, -ValueInf, depth, childNodeCount);

        nodeCount += childNodeCount;

        return maxValue;
    }

    /// <summary>
    /// Performs shallow alpha-beta search without transposition table for leaf-near nodes.
    /// This optimized version reduces overhead when search depth is small.
    /// </summary>
    /// <typeparam name="Phase">The game phase (Midgame or Endgame)</typeparam>
    /// <typeparam name="PassFlag">Indicates if this search follows a pass move</typeparam>
    /// <param name="state">The current game state</param>
    /// <param name="alpha">The alpha bound for alpha-beta pruning</param>
    /// <param name="beta">The beta bound for alpha-beta pruning</param>
    /// <param name="pv">The principal variation being built</param>
    /// <param name="depth">The remaining search depth</param>
    /// <param name="nodeCount">Reference to the node count for statistics</param>
    /// <param name="stopFlag">Reference to the stop flag for search interruption</param>
    /// <returns>The evaluation value of the position</returns>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    unsafe int SearchShallow<Phase, PassFlag>(ref State state, int alpha, int beta, ref PV pv, int depth, ref long nodeCount, ref bool stopFlag)
        where Phase : struct, IGamePhase where PassFlag : struct, IPassFlag
    {
        Debug.Assert(depth >= 0);

        Span<Move> moves = stackalloc Move[Constants.MaxLegalMoves];
        moves = moves[..state.Position.GetNextMoves(ref moves)];
        Span<int> moveScores = stackalloc int[moves.Length];

        if (moves.Length == 0)  // pass
        {
            if (typeof(PassFlag) == typeof(AfterPass))    // Two consecutive passes -> game end
                return DiscDiffToValue(state.Position.DiscDiff);

            state.Pass();

            int pass_value = -SearchShallow<Phase, AfterPass>(ref state, -beta, -alpha, ref pv, depth, ref nodeCount, ref stopFlag);

            state.Pass();

            return pass_value;
        }

        if (depth == 0)
            return Evaluate(ref state);

        int ply = pv.Count;
        pv.Add(BoardCoordinate.Null);   // Store a dummy move to simplify conditional branching.

        InitMoves(ref state.Position, moves);

        var noOrdering = depth <= NoMoveOrderingDepth;
        if (!noOrdering)
        {
            EvaluateMoves<Phase>(ref state, moves, moveScores, depth);
            SortMoves(moves, moveScores);
        }

        var currentPV = new PV();
        int maxValue = -ValueInf;
        var bestMove = BoardCoordinate.Null;
        for (var i = 0; i < moves.Length; i++)
        {
            ref var move = ref moves[i];

            state.Update(ref move);
            currentPV.Add(move.Coord);
            nodeCount++;

            int value = -SearchShallowNext<Phase>(ref state, -beta, -alpha, ref currentPV, depth - 1, ref nodeCount, ref stopFlag);

            state.Undo(ref move);

            // beta cut
            if (value >= beta)
            {
                pv.RemoveUnder(ply);
                pv.AddRange(ref currentPV);
                return value;
            }

            if (value >= maxValue)
            {
                maxValue = value;
                alpha = Math.Max(alpha, value);
                bestMove = move.Coord;

                pv.RemoveUnder(ply);
                pv.AddRange(ref currentPV);
            }

            currentPV.Clear();
        }

        if (AggressivePVSave && maxValue >= alpha)
        {
            ref TTEntry entry = ref _tt.GetEntry(ref state.Position, out var hit);
            if (!hit)
                _tt.SaveAt(ref entry, ref state.Position, bestMove, maxValue, maxValue, depth, nodeCount);
        }

        return maxValue;
    }

    /// <summary>
    /// Creates a complete principal variation list by combining the current PV with
    /// additional moves reconstructed from the transposition table when possible.
    /// </summary>
    /// <param name="pv">The current principal variation</param>
    /// <param name="maxDepth">The maximum depth searched</param>
    /// <returns>A list of moves representing the complete principal variation</returns>
    List<BoardCoordinate> CreatePVList(ref PV pv, int maxDepth)
    {
        var pvList = new List<BoardCoordinate>();
        for (var i = 0; i < pv.Count; i++)
            pvList.Add(pv[i]);

        Debug.Assert(_rootWasSet);

        // When search was cut short by transposition table before reaching leaf nodes,
        // try to reconstruct the remaining PV from the transposition table.
        // Note: This may not always succeed if the TT is small or after long searches.
        if (pv.CutByTT)
        {
            Position pos = _rootPos;
            pv.UpdatePositionAlongPV(ref pos);
            _tt.ProbePV(ref pos, pvList, maxDepth - pv.Count);
        }

        return pvList;
    }

    /// <summary>
    /// Converts a disc count difference to a search value for terminal positions.
    /// </summary>
    /// <param name="discDiff">The disc count difference (positive = win, negative = loss, zero = draw)</param>
    /// <returns>The corresponding search value (ValueInf for win, -ValueInf for loss, 0 for draw)</returns>
    static int DiscDiffToValue(int discDiff)
    {
        if (discDiff == 0)
            return 0;
        return (discDiff > 0) ? ValueInf : -ValueInf;
    }

    /// <summary>
    /// Initializes moves by calculating the flipped discs for each move.
    /// </summary>
    /// <param name="pos">The position to calculate moves for</param>
    /// <param name="moves">The moves to initialize with flip information</param>
    static void InitMoves(ref Position pos, Span<Move> moves)
    {
        for (var i = 0; i < moves.Length; i++)
            pos.CalcFlip(ref moves[i]);
    }

    /// <summary>
    /// Moves the specified move to the front of the move list for move ordering.
    /// This method assumes that the move exists in the list (guaranteed by caller).
    /// </summary>
    /// <param name="moves">The list of moves to reorder</param>
    /// <param name="move">The move to place at the front</param>
    static void PlaceToHead(Span<Move> moves, BoardCoordinate move)
    {
        int i;
        for (i = 0; i < moves.Length; i++)
        {
            if (moves[i].Coord == move)
                break;
        }

        Debug.Assert(i != moves.Length);

        if (i != 0)
            (moves[0], moves[i]) = (moves[i], moves[0]);
    }

    /// <summary>
    /// Sorts moves by their evaluation scores using insertion sort.
    /// Insertion sort is efficient for small arrays typically encountered in move ordering.
    /// </summary>
    /// <param name="moves">The moves to sort</param>
    /// <param name="scores">The corresponding scores for each move</param>
    static void SortMoves(Span<Move> moves, Span<int> scores)
    {
        for (int i = 1; i < scores.Length; i++)
        {
            Move move = moves[i];
            int tmpScore = scores[i];
            if (scores[i - 1] < tmpScore)
            {
                int j = i;
                do
                {
                    moves[j] = moves[j - 1];
                    scores[j] = scores[j - 1];
                    --j;
                } while (j > 0 && scores[j - 1] < tmpScore);

                moves[j] = move;
                scores[j] = tmpScore;
            }
        }
    }

    /// <summary>
    /// Marker interface for game phase types used in generic search methods.
    /// </summary>
    interface IGamePhase { }
    
    /// <summary>
    /// Represents the midgame phase where specific heuristics and evaluation are used.
    /// </summary>
    struct Midgame : IGamePhase { }
    
    /// <summary>
    /// Represents the endgame phase where perfect play algorithms may be employed.
    /// </summary>
    struct Endgame : IGamePhase { }

    /// <summary>
    /// Marker interface for pass flag types used to track consecutive passes.
    /// </summary>
    interface IPassFlag { }
    
    /// <summary>
    /// Indicates that the search is being performed after a pass move.
    /// Two consecutive passes indicate game termination.
    /// </summary>
    struct AfterPass : IPassFlag { }
    
    /// <summary>
    /// Indicates that the search is not being performed after a pass move.
    /// </summary>
    struct NotAfterPass : IPassFlag { }
}