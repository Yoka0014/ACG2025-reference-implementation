using System;
using System.IO;
using System.Threading.Tasks;

using ACG2025_reference_implementation.Utils;
using ACG2025_reference_implementation.Reversi;
using ACG2025_reference_implementation.Evaluation;
using ACG2025_reference_implementation.Search.AlphaBetaPruning;

namespace ACG2025_reference_implementation.Engines;

internal class AlphaBetaPruningEngine : Engine
{
    const int DefaultTTSizeMiB = 512;
    const int TTSizeMibMin = 1;
    const string DefaultValueFuncWeightsPath = "params/value_func_weights.bin";
    const int WaitTimeoutMs = 10000;

    // Depth[level][num_empty_cells]
    static readonly int[][] Depth;

    readonly TimeControl[] _timeControls = [new TimeControl { MainTimeMs = int.MaxValue }, new TimeControl { MainTimeMs = int.MaxValue }];
    int _level = 1;
    ValueFunction _valueFunc;
    Searcher _searcher;
    Task<SearchResult?> _searchTask;

    static AlphaBetaPruningEngine()
    {
        Depth = new int[Constants.NumCells + 1][];
        for (int i = 0; i <= Constants.NumCells; i++)
        {
            Depth[i] = new int[61];
        }

        for (int level = 0; level <= Constants.NumCells; level++)
        {
            if (level >= 0 && level <= 10)
            {
                // Lv.1 ~ Lv.10までは，levelの値と同数の深さを読み，残り level * 2　手は完全読みを行う．
                for (int numEmpties = 0; numEmpties <= 60; numEmpties++)
                    Depth[level][numEmpties] = (numEmpties > level * 2) ? level : numEmpties;
            }
            else if (level >= 11 && level <= 22)
            {
                // Lv.11 ~ Lv.22までは，levelの値と同数の深さを読み，残り22手以下は完全読みを行う．
                for (int numEmpties = 0; numEmpties <= 60; numEmpties++)
                    Depth[level][numEmpties] = (numEmpties > 22) ? level : numEmpties;
            }
            else if (level >= 23 && level <= 30)
            {
                // Lv.23 ~ Lv.30までは，levelの値と同数の深さを読み，残り30手以下は完全読みを行う．
                for (int numEmpties = 0; numEmpties <= 60; numEmpties++)
                    Depth[level][numEmpties] = (numEmpties > 30) ? level : numEmpties;
            }
            else
            {
                // L.31以降は，levelの値と同数の深さを読む．
                for (int numEmpties = 0; numEmpties <= 60; numEmpties++)
                    Depth[level][numEmpties] = (numEmpties > level) ? level : numEmpties;
            }
        }
    }

    public AlphaBetaPruningEngine() : base("AlphaBetaPruningEngine", "0.0", "Yoka0014") => InitOptions();

    void InitOptions()
    {
        Options["tt_size_mib"] = new EngineOption(DefaultTTSizeMiB, min: TTSizeMibMin, long.MaxValue);
        Options["value_func_weights_path"] = new EngineOption(DefaultValueFuncWeightsPath);
        Options["show_search_result_interval_cs"] = new EngineOption(50, 0, long.MaxValue);

        Options["tt_size_mib"].ValueChanged += TTSizeChanged;
        Options["value_func_weights_path"].ValueChanged += ValueFuncWeightsPathChanged;
    }

    public override void Quit()
    {
        if (_searcher is not null && _searcher.IsSearching)
            _searcher.Stop();
    }

    /// <inheritdoc/>
    public override void SetMainTime(DiscColor color, int mainTimeMs) => _timeControls[(int)color].MainTimeMs = mainTimeMs;

    /// <inheritdoc/>
    public override void SetByoyomi(DiscColor color, int byoyomiMs) => _timeControls[(int)color].ByoyomiMs = byoyomiMs;

    /// <inheritdoc/>
    public override void SetByoyomiStones(DiscColor color, int byoyomiStones) => _timeControls[(int)color].ByoyomiStones = byoyomiStones;

    /// <inheritdoc/>
    public override void SetTimeIncrement(DiscColor color, int incMs) => _timeControls[(int)color].IncrementMs = incMs;

    public override void SetLevel(int level) => _level = level;

    public override void Go()
    {
        if (_searcher is null)
        {
            SendErrorMessage("Specify the value function's weights file.");
            return;
        }

        if (!StopIfPondering())
        {
            SendErrorMessage("Cannot stop pondering. There must be some troubles in search thread.");
            return;
        }

        if (Position.CanPass)
        {
            var passMove = new EngineMove { Coord = BoardCoordinate.Pass };
            SendMove(passMove);
            return;
        }

        Span<Move> moves = stackalloc Move[Constants.MaxLegalMoves];
        int moveCount = Position.GetNextMoves(ref moves);
        if (moveCount == 1)
        {
            var singleMove = new EngineMove { Coord = moves[0].Coord };
            SendMove(singleMove);
            return;
        }

        GenerateMove();
    }
    public override void Analyze(int numHints)
    {
        // Note: multi pvが未実装なので，numHintsは無視する．

        if (_searcher is null)
        {
            SendErrorMessage("Specify the value function's weights file.");
            return;
        }

        if (!StopIfPondering())
        {
            SendErrorMessage("Cannot stop pondering. There must be some troubles in search thread.");
            return;
        }

        void OnCompleted(object? sender, SearchResult res)
        {
            var thinkInfo = SearchResultToThinkInfo(res);
            var multiPV = new MultiPV { SearchResultToMultiPVItem(res) };

            SendThinkInfo(thinkInfo);
            SendMultiPV(multiPV);
        }

        _searcher.PVNotificationIntervalMs = Options["show_search_result_interval_cs"] * 10;
        _searchTask = _searcher.SearchAsync(Depth[_level][Position.EmptyCellCount], OnCompleted);
    }

    public override bool StopThinking(int timeoutMs)
    {
        if (_searcher is null || !_searcher.IsSearching)
            return true;

        _searcher.Stop();
        return _searchTask.Wait(timeoutMs);
    }

    protected override bool OnReady()
    {
        string valueFuncWeightsPath = Options["value_func_weights_path"];
        if (!File.Exists(valueFuncWeightsPath))
        {
            SendErrorMessage($"Cannot find evaluator's parameters file: \"" + valueFuncWeightsPath + "\".");
            return false;
        }

        try
        {
            _valueFunc = ValueFunction.LoadFromFile(valueFuncWeightsPath);
        }
        catch (InvalidDataException ex)
        {
            SendErrorMessage($"Cannot load evaluator's parameters. Detail: \"{ex.Message}\"");
            return false;
        }

        var pos = Position;
        long ttSize = Options["tt_size_mib"] * 1024 * 1024;
        _searcher = new Searcher(_valueFunc, ttSize);
        _searcher.SetRoot(ref pos);
        _searcher.SearchResultUpdated += (sender, res) =>
        {
            var thinkInfo = SearchResultToThinkInfo(res);
            var multiPV = new MultiPV { SearchResultToMultiPVItem(res) };

            SendThinkInfo(thinkInfo);
            SendMultiPV(multiPV);
        };

        return true;
    }

    protected override void OnInitializedPosition()
    {
        if (!StopIfPondering())
        {
            SendErrorMessage("Cannot stop pondering. There must be some troubles in search thread.");
            return;
        }

        if (_searcher is not null)
        {
            var pos = Position;
            _searcher.SetRoot(ref pos);
        }
    }

    protected override void OnUpdatedPosition(Move move)
    {
        if (!StopIfPondering())
        {
            SendErrorMessage("Cannot stop pondering. There must be some troubles in search thread.");
            return;
        }

        if (_searcher is not null)
        {
            if (!_searcher.TryUpdateRoot(move.Coord))
            {
                var pos = Position;
                _searcher.SetRoot(ref pos);
            }
        }
    }

    protected override void OnUndonePosition(Move move) => OnInitializedPosition();

    void GenerateMove()
    {
        void OnCompleted(object? sender, SearchResult res)
        {
            var thinkInfo = SearchResultToThinkInfo(res);
            var multiPV = new MultiPV { SearchResultToMultiPVItem(res) };

            SendThinkInfo(thinkInfo);
            SendMultiPV(multiPV);
            
            var move = new EngineMove
            {
                Coord = res.BestMove,
                EvalScore = SearchValueToWinRate(res.SearchValue),
                EvalScoreType = EvalScoreType.WinRate,
                ElapsedMs = res.ElapsedMs
            };
            SendMove(move);
        };

        _searcher.PVNotificationIntervalMs = Options["show_search_result_interval_cs"] * 10;
        _searchTask = _searcher.SearchAsync(Depth[_level][Position.EmptyCellCount], OnCompleted);
    }

    bool StopIfPondering()
    {
        if (_searcher is not null && _searcher.IsSearching)
        {
            _searcher.Stop();
            return _searchTask.Wait(WaitTimeoutMs);
        }
        return true;
    }

    void TTSizeChanged(EngineOption option)
    {
        if (!IsReady)
            return;

        if (!StopIfPondering())
        {
            SendErrorMessage("Cannot stop pondering. There must be some troubles in search thread.");
            return;
        }

        _searcher.TryResizeTranspositionTable(option * 1024L * 1024L);
    }

    void ValueFuncWeightsPathChanged(EngineOption option)
    {
        string path = option;
        if(!File.Exists(path))
        {
            SendErrorMessage($"Cannot find evaluator's parameters file: \"{path}\".");
            return;
        }

        if(!IsReady)
            return;

        OnReady();
    }

    /// <summary>
    /// Converts a search value to a win rate using sigmoid function.
    /// </summary>
    /// <param name="value">The search value to convert</param>
    /// <returns>Win rate as a value between 0.0 and 1.0</returns>
    static double SearchValueToWinRate(int value)
    {
        double logit = (double)value / ValueFunction.OutScale;
        return MathFunctions.StdSigmoid(logit);
    }

    /// <summary>
    /// Converts search results to thinking information.
    /// </summary>
    /// <param name="searchResult">The search result to convert</param>
    /// <returns>Thinking information with evaluation converted to win rate</returns>
    static ThinkInfo SearchResultToThinkInfo(Search.AlphaBetaPruning.SearchResult searchResult)
    {
        return new ThinkInfo(searchResult.PV)
        {
            Depth = searchResult.Depth,
            NodeCount = searchResult.NodeCount,
            Nps = searchResult.NodeCount / (searchResult.ElapsedMs * 1.0e-3),
            EvalScore = SearchValueToWinRate(searchResult.SearchValue)
        };
    }

    /// <summary>
    /// Converts search results to a multi-PV item.
    /// </summary>
    /// <param name="searchResult">The search result to convert</param>
    /// <returns>Multi-PV item with evaluation converted to win rate</returns>
    static MultiPVItem SearchResultToMultiPVItem(Search.AlphaBetaPruning.SearchResult searchResult)
    {
        return new MultiPVItem(searchResult.PV)
        {
            Depth = searchResult.Depth,
            EvalScore = SearchValueToWinRate(searchResult.SearchValue),
            EvalScoreType = EvalScoreType.WinRate
        };
    }
}