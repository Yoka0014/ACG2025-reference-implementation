using System;
using System.IO;
using System.Threading.Tasks;

using ACG2025_reference_implementation.Utils;
using ACG2025_reference_implementation.Reversi;
using ACG2025_reference_implementation.Evaluation;
using ACG2025_reference_implementation.Search.AlphaBetaPruning;

namespace ACG2025_reference_implementation.Engines;

/// <summary>
/// Alpha-beta pruning engine for Reversi game.
/// This engine uses alpha-beta search with transposition table, move ordering,
/// and a n-tuple-based evaluation function for strong gameplay.
/// </summary>
internal class AlphaBetaPruningEngine : Engine
{
    /// <summary>Default size of transposition table in mebibytes</summary>
    const int DefaultTTSizeMiB = 512;
    /// <summary>Minimum allowed size of transposition table in mebibytes</summary>
    const int TTSizeMibMin = 1;
    /// <summary>Default path to the value function weights file</summary>
    const string DefaultValueFuncWeightsPath = "params/value_func_weights.bin";
    /// <summary>Timeout for waiting search operations in milliseconds</summary>
    const int WaitTimeoutMs = 10000;

    /// <summary>
    /// Search depth configuration array indexed by level and number of empty cells.
    /// Determines search depth based on engine level and game phase.
    /// </summary>
    static readonly int[][] Depth;

    /// <summary>Time control settings for both players (Black and White)</summary>
    readonly TimeControl[] _timeControls = [new TimeControl { MainTimeMs = int.MaxValue }, new TimeControl { MainTimeMs = int.MaxValue }];
    /// <summary>Current engine thinking level</summary>
    int _level = 1;
    /// <summary>Neural network-based evaluation function</summary>
    ValueFunction? _valueFunc;
    /// <summary>Alpha-beta search algorithm implementation</summary>
    Searcher? _searcher;
    /// <summary>Currently running search task</summary>
    Task<SearchResult?>? _searchTask;

    /// <summary>
    /// Static constructor that initializes the depth configuration table.
    /// Sets up search depths for different engine levels and game phases.
    /// </summary>
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
                // For levels 1-10: search to depth = level, with perfect search for remaining level * 2 moves
                for (int numEmpties = 0; numEmpties <= 60; numEmpties++)
                    Depth[level][numEmpties] = (numEmpties > level * 2) ? level : numEmpties;
            }
            else if (level >= 11 && level <= 22)
            {
                // For levels 11-22: search to depth = level, with perfect search for remaining 22 moves or fewer
                for (int numEmpties = 0; numEmpties <= 60; numEmpties++)
                    Depth[level][numEmpties] = (numEmpties > 22) ? level : numEmpties;
            }
            else if (level >= 23 && level <= 30)
            {
                // For levels 23-30: search to depth = level, with perfect search for remaining 30 moves or fewer
                for (int numEmpties = 0; numEmpties <= 60; numEmpties++)
                    Depth[level][numEmpties] = (numEmpties > 30) ? level : numEmpties;
            }
            else
            {
                // For level 31 and above: search to depth = level
                for (int numEmpties = 0; numEmpties <= 60; numEmpties++)
                    Depth[level][numEmpties] = (numEmpties > level) ? level : numEmpties;
            }
        }
    }

    /// <summary>
    /// Initializes a new instance of the AlphaBetaPruningEngine class.
    /// Sets up engine metadata and configuration options.
    /// </summary>
    public AlphaBetaPruningEngine() : base("AlphaBetaPruningEngine", "0.0", "Yoka0014") => InitOptions();

    /// <summary>
    /// Initializes engine configuration options and sets up event handlers.
    /// </summary>
    void InitOptions()
    {
        Options["tt_size_mib"] = new EngineOption(DefaultTTSizeMiB, min: TTSizeMibMin, long.MaxValue);
        Options["value_func_weights_path"] = new EngineOption(DefaultValueFuncWeightsPath);
        Options["show_search_result_interval_cs"] = new EngineOption(50, 0, long.MaxValue);

        Options["tt_size_mib"].ValueChanged += TTSizeChanged;
        Options["value_func_weights_path"].ValueChanged += ValueFuncWeightsPathChanged;
    }

    /// <summary>
    /// Terminates the engine and stops any ongoing search operations.
    /// </summary>
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

    /// <summary>
    /// Sets the engine's thinking level, which determines search depth and strength.
    /// </summary>
    /// <param name="level">The level to set (higher values result in stronger play)</param>
    public override void SetLevel(int level) => _level = level;

    /// <summary>
    /// Starts the engine thinking process to generate a move for the current position.
    /// Handles special cases like forced passes and single legal moves before starting search.
    /// </summary>
    public override void Go()
    {
        if (_searcher is null)
        {
            SendErrorMessage("Value function weights file must be specified.");
            return;
        }

        if (!StopIfPondering())
        {
            SendErrorMessage("Cannot stop current search. There may be issues with the search thread.");
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
    /// <summary>
    /// Starts position analysis to provide multiple candidate moves with evaluations.
    /// Currently only provides the principal variation (single best line).
    /// </summary>
    /// <param name="numHints">Number of candidate moves to analyze (currently ignored as multi-PV is not implemented)</param>
    public override void Analyze(int numHints)
    {
        if (_searcher is null)
        {
            SendErrorMessage("Value function weights file must be specified.");
            return;
        }

        if (!StopIfPondering())
        {
            SendErrorMessage("Cannot stop current search. There may be issues with the search thread.");
            return;
        }

        void OnCompleted(object? sender, SearchResult res)
        {
            var thinkInfo = SearchResultToThinkInfo(res);
            var multiPV = new MultiPV { SearchResultToMultiPVItem(res) };

            SendThinkInfo(thinkInfo);
            SendMultiPV(multiPV);
        }

        _searcher!.PVNotificationIntervalMs = Options["show_search_result_interval_cs"] * 10;
        _searchTask = _searcher.SearchAsync(Depth[_level][Position.EmptyCellCount], OnCompleted);
    }

    /// <summary>
    /// Stops the current thinking process with a specified timeout.
    /// </summary>
    /// <param name="timeoutMs">Maximum time to wait for the search to stop in milliseconds</param>
    /// <returns>True if the search was stopped successfully within the timeout, false otherwise</returns>
    public override bool StopThinking(int timeoutMs)
    {
        if (_searcher is null || !_searcher.IsSearching)
            return true;

        _searcher.Stop();
        return _searchTask?.Wait(timeoutMs) ?? true;
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
            SendErrorMessage($"Cannot load value function weights. Details: \"{ex.Message}\"");
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

    /// <summary>
    /// Called when the game position is initialized to the starting state.
    /// Updates the search tree root to match the new position.
    /// </summary>
    protected override void OnInitializedPosition()
    {
        if (!StopIfPondering())
        {
            SendErrorMessage("Cannot stop current search. There may be issues with the search thread.");
            return;
        }

        if (_searcher is not null)
        {
            var pos = Position;
            _searcher.SetRoot(ref pos);
        }
    }

    /// <summary>
    /// Called when a move is made on the board.
    /// Attempts to update the search tree incrementally, falling back to full reset if necessary.
    /// </summary>
    /// <param name="move">The move that was made</param>
    protected override void OnUpdatedPosition(Move move)
    {
        if (!StopIfPondering())
        {
            SendErrorMessage("Cannot stop current search. There may be issues with the search thread.");
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

    /// <summary>
    /// Called when a move is undone from the board.
    /// Resets the search tree to match the current position.
    /// </summary>
    /// <param name="move">The move that was undone</param>
    protected override void OnUndonePosition(Move move) => OnInitializedPosition();

    /// <summary>
    /// Initiates the search process to generate the best move for the current position.
    /// Sets up search completion callback and starts asynchronous search.
    /// </summary>
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

        _searcher!.PVNotificationIntervalMs = Options["show_search_result_interval_cs"] * 10;
        _searchTask = _searcher.SearchAsync(Depth[_level][Position.EmptyCellCount], OnCompleted);
    }

    /// <summary>
    /// Stops any ongoing search (pondering) if one is in progress.
    /// </summary>
    /// <returns>True if no search was running or if the search was stopped successfully, false if timeout occurred</returns>
    bool StopIfPondering()
    {
        if (_searcher is not null && _searcher.IsSearching)
        {
            _searcher.Stop();
            return _searchTask?.Wait(WaitTimeoutMs) ?? true;
        }
        return true;
    }

    /// <summary>
    /// Event handler called when the transposition table size option is changed.
    /// Resizes the transposition table if the engine is ready and not currently searching.
    /// </summary>
    /// <param name="option">The engine option containing the new table size in MiB</param>
    void TTSizeChanged(EngineOption option)
    {
        if (!IsReady)
            return;

        if (!StopIfPondering())
        {
            SendErrorMessage("Cannot stop current search. There may be issues with the search thread.");
            return;
        }

        _searcher?.TryResizeTranspositionTable(option * 1024L * 1024L);
    }

    /// <summary>
    /// Event handler called when the value function weights file path option is changed.
    /// Reloads the value function if the engine is ready and the file exists.
    /// </summary>
    /// <param name="option">The engine option containing the new file path</param>
    void ValueFuncWeightsPathChanged(EngineOption option)
    {
        string path = option;
        if(!File.Exists(path))
        {
            SendErrorMessage($"Cannot find value function weights file: \"{path}\".");
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