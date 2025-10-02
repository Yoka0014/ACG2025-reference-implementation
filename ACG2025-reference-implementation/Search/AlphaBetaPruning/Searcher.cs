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

internal class SearchResult(IEnumerable<BoardCoordinate> pv)
{
    public BoardCoordinate BestMove { get; init; }
    public int SearchValue { get; init; }
    public int Depth { get; init; }
    public long NodeCount { get; init; }
    public int ElapsedMs { get; init; }
    public ReadOnlyCollection<BoardCoordinate> PV => new(_pv);

    readonly List<BoardCoordinate> _pv = [.. pv];
};

struct PV
{
    public int Count { get; private set; }
    public bool CutByTT { get; set; }

    Moves _moves;

    public readonly BoardCoordinate this[int idx]
    {
        get
        {
            if (idx < 0 || idx >= Count)
                throw new IndexOutOfRangeException();

            return _moves[idx];
        }
    }

    public void Clear()
    {
        Count = 0;
        CutByTT = false;
    }

    public readonly void CopyTo(ref PV dest)
    {
        dest.Count = Count;
        dest.CutByTT = CutByTT;

        for (var i = 0; i < Count; i++)
            dest._moves[i] = _moves[i];
    }

    public readonly bool Contains(BoardCoordinate move)
    {
        for (var i = 0; i < Count; i++)
        {
            if (_moves[i] == move)
                return true;
        }
        return false;
    }

    public void Add(BoardCoordinate move) => _moves[Count++] = move;

    public void AddRange(ref PV pv)
    {
        for (var i = 0; i < pv.Count; i++)
            Add(pv._moves[i]);
        CutByTT = pv.CutByTT;
    }

    public void RemoveUnder(int start)
    {
        Debug.Assert(0 <= start && start < Count);

        Count -= Count - start;
        CutByTT = false;
    }

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

    public override readonly string ToString()
    {
        var sb = new StringBuilder();
        for (var i = 0; i < Count; i++)
            sb.Append(Utils.Coordinate1DTo2D(_moves[i]));
        return sb.ToString();
    }

    [InlineArray(Constants.NumInitialCells)]
    struct Moves
    {
        BoardCoordinate _coord;

#pragma warning disable CS9181 // アクセスの高速化のために独自のインデクサを用いるので警告を抑制する．
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

internal class Searcher(ValueFunction valueFunc, long ttSizeBytes)
{
    public const int ValueInf = short.MaxValue;
    public const int ValueInvalid = int.MaxValue;
    public const int DepthMin = 4;
    public const int EndgameDepth = 15;


    // 序中盤において，残りの探索深度がこの値以下になったら高速な探索に切り替える．
    const int MidgameShallowDepth = 4;

    // 終盤において，残りの探索深度がこの値以下になったら高速な探索に切り替える．
    const int EndgameShallowDepth = 7;

    // 終局までの残り手数がこの値以下であれば反復深化を行わない．
    const int MateSearchDepth = 10;

    // 残りの探索深度がこの値以下になったらムーブオーダリングを行わない．
    const int NoMoveOrderingDepth = 2;

    // ムーブオーダリングのときに，局面評価関数を用いるかどうか．
    const bool UseEvaluatorForMoveOrdering = true;

    // 残り探索深度がこの値のノードにおいて，探索中断フラグの確認を行う．
    const int SuspendFlagCheckDepth = 5;

    // 置換表にできる限り最善応手を残すか．
    // これをtrueにすると，葉ノードに近いノードにおいても，最善手を記録するためPVを復元しやすくなる．
    // ただし，強さには影響しない．むしろ置換表アクセスの時間ロスが発生する．
    public bool AggressivePVSave { get; set; } = true;

    public int PVNotificationIntervalMs { get; set; } = 100;

    public event EventHandler<SearchResult> SearchResultUpdated = delegate { };

    public long NodeCount => _nodeCount;
    public bool IsSearching => _isSearching;

    readonly ValueFunction _valueFunc = valueFunc;
    TranspositionTable _tt = new(ttSizeBytes);
    Position? _rootPos;
    long _nodeCount = 0;
    volatile bool _isSearching = false;
    volatile bool _suspendFlag = false;

    public bool TryResizeTranspositionTable(long ttSizeBytes)
    {
        if (IsSearching)
            return false;
        _tt = new TranspositionTable(ttSizeBytes);
        return true;
    }

    public void SetRoot(ref Position pos)
    {
        if (_rootPos.HasValue)
            _tt.Clear();
        _rootPos = pos;
    }

    public bool TryUpdateRoot(BoardCoordinate moveCoord)
    {
        if (!_rootPos.HasValue)
            throw new InvalidOperationException("Root position has not been set.");

        if (!_rootPos.Value.IsLegalMoveAt(moveCoord))
            return false;

        var move = new Move(moveCoord);
        _rootPos.Value.CalcFlip(ref move);
        UpdateRoot(ref move);
        return true;
    }

    public void UpdateRoot(ref Move move)
    {
        if (!_rootPos.HasValue)
            throw new InvalidOperationException("Root position has not been set.");

        if (move.Coord == BoardCoordinate.Pass)
            _rootPos.Value.Pass();
        else
            _rootPos.Value.Update(ref move);

        _tt.IncGen();
    }

    public void PassRoot()
    {
        var move = new Move(BoardCoordinate.Pass);
        UpdateRoot(ref move);
    }

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

    public SearchResult Search(int maxDepth)
    {
        if (!_rootPos.HasValue)
            throw new InvalidOperationException("Root position has not been set.");

        _suspendFlag = false;
        _isSearching = true;
        _nodeCount = 0;

        SearchResult res;
        var pv = new PV();
        int depth;
        var rootPos = _rootPos.Value;
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

            Debug.Assert(hit);    // ルート局面の価値は，最後に置換表に記録されるので必ずヒットするはず．
            Debug.Assert(entry.HasBestMove);   // また，ルート局面の置換表エントリには，必ず最善手記録されるはず．

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
        
        int idMax = maxDepth;
        if(idMax >= state.Position.EmptyCellCount)
            idMax = Math.Max(0, state.Position.EmptyCellCount - MateSearchDepth);

        depth = (maxDepth % 2 == 0) ? DepthMin : DepthMin + 1;

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
                break;;

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

    int Evaluate(ref State state) => _valueFunc.F(state.FeatureVector);

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
                    return SearchWithTT<Phase, AfterPass>(ref state, alpha, beta, ref pv, depth, ref nodeCount, ref stopFlag);
                else
                    return SearchWithTT<Endgame, AfterPass>(ref state, alpha, beta, ref pv, depth, ref nodeCount, ref stopFlag);
            }
            else
            {
                return SearchShallow<Phase, AfterPass>(ref state, alpha, beta, ref pv, depth, ref nodeCount, ref stopFlag);
            }
        }
        else
        {
            if (depth > EndgameShallowDepth)
                return SearchWithTT<Endgame, AfterPass>(ref state, alpha, beta, ref pv, depth, ref nodeCount, ref stopFlag);
            else
                return SearchShallow<Endgame, AfterPass>(ref state, alpha, beta, ref pv, depth, ref nodeCount, ref stopFlag);
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
            // 以前に探索済みの局面の場合は，置換表エントリから評価値の範囲を取得．
            int lower = entry.Lower;
            int upper = entry.Upper;

            // state.pos_の真の評価値をvとすると，lower <= v <= upperを満たす．
            // そのため，upper <= alphaならalpha cutを，beta <= lowerならbeta cutができる．
            // また，lower == upperのときは，v == lower == upperなので，lower（もしくはupper）をそのまま返せば良い．

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

            // 探索窓を狭める．
            alpha = Math.Max(alpha, lower);
            beta = Math.Min(beta, upper);
        }

        Span<Move> moves = stackalloc Move[Constants.MaxLegalMoves];
        moves = moves[..state.Position.GetNextMoves(ref moves)];
        Span<int> moveScores = stackalloc int[moves.Length];

        if (moves.Length == 0)  // pass
        {
            if (typeof(PassFlag) == typeof(AfterPass))   // 2連続パス -> 終局
                return DiscDiffToValue(state.Position.DiscDiff);

            state.Pass();

            int pass_value = -SearchWithTT<Phase, AfterPass>(ref state, -beta, -alpha, ref pv, depth, ref nodeCount, ref stopFlag);

            state.Pass();

            return pass_value;
        }

        if (depth == 0)  // 葉ノード
            return Evaluate(ref state);

        // 以降，内部ノードに対する処理．

        int ply = pv.Count;

        // 裏返る石の計算．
        InitMoves(ref state.Position, moves);

        if (hit && entry.HasBestMove)
        {
            // 置換表に前回の探索の最善手が記録されている場合，それを先頭に配置．
            PlaceToHead(moves, entry.Move);
        }
        else
        {
            // 全ての手にスコアリングを行う．
            EvaluateMoves<Phase>(ref state, moves, moveScores, depth);
            SortMoves(moves, moveScores);
        }

        // 着手リストの先頭が最善手であると仮定．
        ref var bestMove = ref moves[0];
        int maxValue, value, a = alpha;
        long childNodeCount = 0;

        state.Update(ref bestMove);
        pv.Add(bestMove.Coord);

        // 最善手候補を現在の探索窓で探索．
        maxValue = value = -SearchNext<Phase>(ref state, -beta, -a, ref pv, depth - 1, ref childNodeCount, ref stopFlag);

        state.Undo(ref bestMove);

        if (stopFlag)
            return ValueInvalid;

        // beta cut
        if (value >= beta)
        {
            // beta cutが起きたので，この局面の評価値は少なくともvalue以上であることを記録する．
            // value == ValueInfであれば勝ち確定なので，best_moveも記録する．
            if (value != ValueInf)
                _tt.SaveAt(ref entry, ref state.Position, value, ValueInf, depth, childNodeCount);
            else
                _tt.SaveAt(ref entry, ref state.Position, bestMove.Coord, ValueInf, ValueInf, depth, childNodeCount);

            nodeCount += childNodeCount;
            return value;
        }

        // この局面の評価値は少なくともvalue以上なのでalpha値を更新する．
        if (value > a)
            a = value;

        // 置換表に前回の探索の最善手が記録されていたが，beta cutが発生しなかった場合，
        // 残りの着手のスコアリングを行う．
        var offset = 0;
        if (hit && entry.HasBestMove && moves.Length > 1)
        {
            EvaluateMoves<Phase>(ref state, moves[1..], moveScores[1..], depth);
            SortMoves(moves[1..], moveScores[1..]);
            offset = 1;
        }

        // 残りの手を探索．
        var currentPV = new PV();
        for (var i = offset; i < moves.Length; i++)
        {
            ref Move move = ref moves[i];
            state.Update(ref move);

            // NWSで現在のalpha値を超えるかどうか確認．
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

            // 現在のalpha値を超えることが判明したので，現在の探索窓で再探索．
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

            // 最善手の更新
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

        if (maxValue > alpha)   // 探索窓内で評価値が判明 -> 真の評価値, max_value >= alphaとはしない(子孫ノードでbeta cutが起きてalphaが返ってきたケースまで含まれてしまう)
            _tt.SaveAt(ref entry, ref state.Position, bestMove.Coord, maxValue, maxValue, depth, childNodeCount);
        else if (maxValue != -ValueInf)    // alphaを下回ったので，この局面の評価値は少なくともmax_value以下であることが判明．
            _tt.SaveAt(ref entry, ref state.Position, -ValueInf, maxValue, depth, childNodeCount);
        else    // max_valueが-kValueInfなら，この局面は負け確定であることが判明．
            _tt.SaveAt(ref entry, ref state.Position, bestMove.Coord, -ValueInf, -ValueInf, depth, childNodeCount);

        nodeCount += childNodeCount;

        return maxValue;
    }

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
            if (typeof(PassFlag) == typeof(AfterPass))    // 2連続パス -> 終局
                return DiscDiffToValue(state.Position.DiscDiff);

            state.Pass();

            int pass_value = -SearchShallow<Phase, AfterPass>(ref state, -beta, -alpha, ref pv, depth, ref nodeCount, ref stopFlag);

            state.Pass();

            return pass_value;
        }

        if (depth == 0)
            return Evaluate(ref state);

        int ply = pv.Count;
        pv.Add(BoardCoordinate.Null);   // 条件分岐を簡略化するためにダミームーブを格納。

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

    List<BoardCoordinate> CreatePVList(ref PV pv, int maxDepth)
    {
        var pvList = new List<BoardCoordinate>();
        for (var i = 0; i < pv.Count; i++)
            pvList.Add(pv[i]);

        Debug.Assert(_rootPos is not null);

        // 置換表によって，葉ノードに到達する前に探索が切り上げられている場合は，置換表から残りのPVを探す．
        // ただし，置換表が小さい場合や長時間探索させた後だと，PVを最後まで復元できない場合がある．
        if (pv.CutByTT)
        {
            Position pos = _rootPos.Value;
            pv.UpdatePositionAlongPV(ref pos);
            _tt.ProbePV(ref pos, pvList, maxDepth - pv.Count);
        }

        return pvList;
    }

    static int DiscDiffToValue(int discDiff)
    {
        if (discDiff == 0)
            return 0;
        return (discDiff > 0) ? ValueInf : -ValueInf;
    }

    static void InitMoves(ref Position pos, Span<Move> moves)
    {
        for (var i = 0; i < moves.Length; i++)
            pos.CalcFlip(ref moves[i]);
    }

    static void PlaceToHead(Span<Move> moves, BoardCoordinate move)
    {
        int i;
        for (i = 0; i < moves.Length; i++)
        {
            if (moves[i].Coord == move)
                break;
        }

        // このメソッドは、movesにmoveが含まれることを呼び出し側が保証していることを前提としている。
        Debug.Assert(i != moves.Length);

        if (i != 0)
            (moves[0], moves[i]) = (moves[i], moves[0]);
    }

    static void SortMoves(Span<Move> moves, Span<int> scores)
    {
        // 要素数が少ない場合は挿入ソートが高速に動作する。
        for (int i = 1; i < scores.Length; i++)
        {
            Move move = moves[i];
            int tmpScore = scores[i];
            if (scores[i - 1] > tmpScore)
            {
                int j = i;
                do
                {
                    moves[j] = moves[j - 1];
                    scores[j] = scores[j - 1];
                    --j;
                } while (j > 0 && scores[j - 1] > tmpScore);

                moves[j] = move;
                scores[j] = tmpScore;
            }
        }
    }

    interface IGamePhase { }
    struct Midgame : IGamePhase { }
    struct Endgame : IGamePhase { }

    interface IPassFlag { }
    struct AfterPass : IPassFlag { }
    struct NotAfterPass : IPassFlag { }
}