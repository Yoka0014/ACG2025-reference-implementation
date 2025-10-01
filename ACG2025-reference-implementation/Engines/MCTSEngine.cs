namespace ACG2025_reference_implementation.Engines;

using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;

using ACG2025_reference_implementation.Utils;
using ACG2025_reference_implementation.Reversi;
using ACG2025_reference_implementation.Evaluation;
using ACG2025_reference_implementation.Search.MCTS;

/// <summary>
/// Monte Carlo Tree Search (MCTS) engine using PUCT algorithm for Reversi.
/// This engine uses a n-Tuple-based value function to evaluate positions
/// and performs tree search using Monte Carlo simulations guided by PUCT selection.
/// </summary>
internal class MCTSEngine : Engine
{
    /// <summary>
    /// Default value function weights file path
    /// </summary>
    const string DefaultValueFuncWeightsPath = "params/value_func_weights.bin";

    /// <summary>Stream writer for logging engine thoughts and search information</summary>
    StreamWriter _logger;

    /// <summary>The PUCT searcher instance that performs Monte Carlo tree search</summary>
    PUCTSearcher? _searcher;
    /// <summary>Current search task for asynchronous search operations</summary>
    Task? _searchTask;
    /// <summary>Random number generator for stochastic move selection</summary>
    Random _rand;

    /// <summary>Time control settings for both players (Black and White)</summary>
    readonly TimeControl[] _timeControls = [new TimeControl(), new TimeControl()];

    /// <summary>
    /// Initializes a new instance of MCTSEngine with no log file.
    /// </summary>
    public MCTSEngine() : this(string.Empty) { }

    /// <summary>
    /// Initializes a new instance of MCTSEngine with the specified log file.
    /// </summary>
    /// <param name="logFilePath">Path to the log file for engine thoughts. Empty string disables logging.</param>
    public MCTSEngine(string logFilePath) : base("PUCTEngine", "0.0", "Yoka0014")
    {
        EvalScoreMin = 0.0;
        EvalScoreMax = 1.0;

        if (string.IsNullOrEmpty(logFilePath))
            _logger = new StreamWriter(Stream.Null);
        else
            _logger = new StreamWriter(logFilePath);

        _rand = new Random(Random.Shared.Next());

        InitOptions();
    }

    /// <summary>
    /// Initializes engine configuration options with default values and sets up event handlers.
    /// </summary>
    void InitOptions()
    {
        Options["num_simulations"] = new EngineOption(800, min: 0);
        Options["num_stochastic_moves"] = new EngineOption(0, min: 0, max: Constants.NumInitialCells);
        Options["reuse_subtree"] = new EngineOption(true);
        Options["num_threads"] = new EngineOption(Environment.ProcessorCount, min: 1, max: Environment.ProcessorCount);
        Options["show_search_result_interval_cs"] = new EngineOption(100, min: 0);
        Options["thought_log_path"] = new EngineOption("thought.log");
        Options["show_log"] = new EngineOption(false);
        Options["value_func_weights_path"] = new EngineOption(DefaultValueFuncWeightsPath);

        Options["value_func_weights_path"].ValueChanged += OnValueFuncWeightsPathSpecified;
        Options["thought_log_path"].ValueChanged += OnThoughtLogPathChanged;
        Options["num_threads"].ValueChanged += OnNumThreadsChanged;
    }

    /// <summary>
    /// Terminates the engine, stops any ongoing search, and releases resources.
    /// </summary>
    public override void Quit()
    {
        if (_searcher is not null && _searcher.IsSearching)
            _searcher.SendStopSearchSignal();
        _logger?.Dispose();
    }

    /// <inheritdoc/>
    public override void SetMainTime(DiscColor color, int mainTimeMs)
    {
        ref TimeControl timeControl = ref _timeControls[(int)color];
        timeControl.MainTimeMs = mainTimeMs;
    }

    /// <inheritdoc/>
    public override void SetByoyomi(DiscColor color, int byoyomiMs) => _timeControls[(int)color].ByoyomiMs = byoyomiMs;

    /// <inheritdoc/>
    public override void SetByoyomiStones(DiscColor color, int byoyomiStones) => _timeControls[(int)color].ByoyomiStones = byoyomiStones;

    /// <inheritdoc/>
    public override void SetTimeIncrement(DiscColor color, int incMs) => _timeControls[(int)color].IncrementMs = incMs;

    /// <inheritdoc/>
    public override void SetLevel(int level)
    {
        Options["num_simulations"].CurrentValue = (128L << (level - 1)).ToString();
        SendTextMessage($"Number of simulations set to {(long)Options["num_simulations"]}");
    }

    /// <summary>
    /// Starts the engine to find and play the best move in the current position.
    /// Handles automatic pass moves and single legal move scenarios efficiently.
    /// </summary>
    public override void Go()
    {
        StopIfPondering();

        if (Position.CanPass)
        {
            SendMove(new EngineMove(BoardCoordinate.Pass));
            return;
        }

        var moves = Position.EnumerateNextMoves();
        if (moves.Count() == 1)
        {
            SendMove(new EngineMove(moves.First()));
            return;
        }

        GenerateMove();
    }

    /// <summary>
    /// Called when the position is initialized. Clears the search tree and sets up the root position.
    /// </summary>
    protected override void OnInitializedPosition()
    {
        if (_searcher is null)
        {
            SendErrorMessage("Weights of the value function were not loaded.");
            return;
        }

        StopIfPondering();

        var pos = Position;
        _searcher.SetRootPosition(ref pos);
        WriteLog("Tree was cleared.\n");
    }

    /// <summary>
    /// Called when a move is made on the board. Updates the search tree by transitioning to the child state
    /// or clears the tree if subtree reuse is disabled or the transition is not possible.
    /// </summary>
    /// <param name="move">The move that was made</param>
    protected override void OnUpdatedPosition(Move move)
    {
        if (_searcher is null)
        {
            SendErrorMessage("Weights of the value function were not loaded.");
            return;
        }

        StopIfPondering();

        if (!Options["reuse_subtree"] || !_searcher.TransitionRootStateToChildState(move.Coord))
        {
            var pos = Position;
            _searcher.SetRootPosition(ref pos);
            WriteLog("Tree was cleared.\n");
        }
        else
            WriteLog("Tree was updated.\n");
    }

    /// <summary>
    /// Called when a move is undone. Clears the search tree as the previous tree state cannot be restored.
    /// </summary>
    /// <param name="move">The move that was undone</param>
    protected override void OnUndonePosition(Move move)
    {
        if (_searcher is null)
        {
            SendErrorMessage("Weights of the value function were not loaded.");
            return;
        }

        StopIfPondering();

        var pos = Position;
        _searcher.SetRootPosition(ref pos);
        WriteLog("Undo.\n");
        WriteLog("Tree was cleared.\n");
    }

    /// <summary>
    /// Starts analysis mode to provide continuous evaluation of the current position.
    /// Runs indefinitely until stopped, providing multi-PV analysis results.
    /// </summary>
    /// <param name="numMoves">Number of candidate moves to analyze (currently unused)</param>
    public override void Analyze(int numMoves)
    {
        StopIfPondering();

        if (Position.CanPass)
        {
            EndAnalysis();
            return;
        }

        AnalyzeMoves();
    }

    /// <summary>
    /// Stops the current search operation with a timeout.
    /// </summary>
    /// <param name="timeoutMs">Maximum time to wait for the search to stop, in milliseconds</param>
    /// <returns>True if the search was stopped within the timeout, false otherwise</returns>
    public override bool StopThinking(int timeoutMs)
    {
        if (_searcher is null || _searchTask is null)
            return true;

        WriteLog("Received stop signal.\n");

        _searcher.SendStopSearchSignal();
        return _searchTask.Wait(timeoutMs);
    }

    /// <summary>
    /// Initializes the engine by loading the value function weights and setting up the search tree.
    /// </summary>
    /// <returns>True if initialization succeeded, false if the weights file could not be loaded</returns>
    protected override bool OnReady()
    {
        string valueFuncWeightsPath = Options["value_func_weights_path"];
        if (!File.Exists(valueFuncWeightsPath))
        {
            SendErrorMessage($"Cannot find value func weights file: \"{valueFuncWeightsPath}\".");
            return false;
        }

        string logPath = Options["thought_log_path"];
        if (!string.IsNullOrEmpty(logPath))
            _logger = new StreamWriter(logPath);
        else
            _logger = new StreamWriter(Stream.Null);

        try
        {
            InitTree(ValueFunction.LoadFromFile(valueFuncWeightsPath));
        }
        catch (InvalidDataException ex)
        {
            SendErrorMessage(ex.Message);
            return false;
        }

        return true;
    }

    /// <summary>
    /// Initializes the PUCT searcher with the given value function and sets up event handlers.
    /// </summary>
    /// <param name="valueFunc">The value function to use for position evaluation</param>
    void InitTree(ValueFunction valueFunc)
    {
        _searcher = new PUCTSearcher(valueFunc);
        var pos = Position;
        _searcher.SetRootPosition(ref pos);
        _searcher.SearchResultUpdated += (s, e) => SendSearchResult(e);
        _searcher.NumThreads = Options["num_threads"];
    }

    /// <summary>
    /// Stops any ongoing search (pondering) and waits for it to complete.
    /// Logs the final search results if available.
    /// </summary>
    void StopIfPondering()
    {
        if (_searcher is not null && _searchTask is not null && !_searchTask.IsCompleted)
        {
            _searcher.SendStopSearchSignal();
            WriteLog("stop pondering.\n\n");

            _searchTask.Wait();

            SearchResult? searchResult;
            if ((searchResult = _searcher.GetSearchResult()) is not null)
                WriteLog(SearchResultToString(searchResult));
        }
    }

    /// <summary>
    /// Generates a move by performing Monte Carlo tree search with time management.
    /// Uses a simple time allocation strategy: remaining main time divided by remaining moves plus byoyomi.
    /// </summary>
    void GenerateMove()
    {
        Debug.Assert(_searcher is not null);
        Debug.Assert(Position.EmptyCellCount != 0);

        WriteLog("Start search.\n");

        _searcher.SearchResultUpdateIntervalCs = Options["show_search_result_interval_cs"];

        // Currently uses a simple time management strategy: main time divided by remaining moves plus byoyomi.
        // This may need improvement for situations requiring strict time management.
        var timeControl = _timeControls[(int)Position.SideToMove];
        var mainTimeCs = Math.Max(timeControl.MainTimeMs / 10, 1) / Position.EmptyCellCount;
        var byoyomiStones = Math.Max(timeControl.ByoyomiStones, 1);
        var byoyomiCs = Math.Max(timeControl.ByoyomiMs / byoyomiStones / 10, 1);
        _searchTask = _searcher.SearchAsync(Options["num_simulations"], mainTimeCs + byoyomiCs, OnCompleted);

        void OnCompleted()
        {
            WriteLog($"End search.\n");

            var searchResult = _searcher.GetSearchResult();

            if (searchResult is null)
                return;

            WriteLog(SearchResultToString(searchResult));
            SendMove(SelectMove(searchResult));
        }
    }

    /// <summary>
    /// Starts continuous analysis of the current position, running indefinitely until stopped.
    /// Provides detailed search results and multi-PV analysis.
    /// </summary>
    void AnalyzeMoves()
    {
        Debug.Assert(_searcher is not null);

        WriteLog("Start search.\n");

        _searcher.SearchResultUpdateIntervalCs = Options["show_search_result_interval_cs"];
        long numSimulations = Options["num_simulations"];
        _searchTask = _searcher.SearchAsync(numSimulations, int.MaxValue / 10, searchEndCallback);

        void searchEndCallback()
        {
            WriteLog("End Search.\n");

            var searchResult = _searcher.GetSearchResult();

            if (searchResult is not null)
            {
                WriteLog(SearchResultToString(searchResult));
                SendSearchResult(searchResult);
            }

            EndAnalysis();
        }
    }

    /// <summary>
    /// Selects the best move from search results, applying stochastic selection for early moves if configured.
    /// For later moves, always selects the move with the highest visit count.
    /// </summary>
    /// <param name="searchResult">The search results containing candidate moves and their evaluations</param>
    /// <returns>The selected engine move with evaluation information</returns>
    EngineMove SelectMove(SearchResult searchResult)
    {
        var childEvals = searchResult.ChildEvals;
        var selectedIdx = 0;
        var moveNum = Constants.NumInitialCells - Position.EmptyCellCount + 1;
        if (moveNum <= Options["num_stochastic_moves"])
            selectedIdx = _rand.Sample(childEvals.ToArray().Select(x => x.Effort).ToArray());
        var selected = childEvals[selectedIdx];
        return new EngineMove(selected.Move, selected.ExpectedReward, EvalScoreType.WinRate, _searcher?.SearchElapsedMs);
    }

    /// <summary>
    /// Sends search results to event listeners and logs the results.
    /// Converts search results to thinking information and multi-PV format.
    /// </summary>
    /// <param name="searchResult">The search results to send, or null if no results available</param>
    void SendSearchResult(SearchResult? searchResult)
    {
        if (searchResult is null)
            return;

        SendThinkInfo(CreateThinkInfo(searchResult));
        SendMultiPV(CreateMultiPV(searchResult));

        WriteLog(SearchResultToString(searchResult));
        WriteLog("\n");
    }

    /// <summary>
    /// Creates thinking information from search results for the best move.
    /// </summary>
    /// <param name="searchResult">The search results to convert</param>
    /// <returns>Thinking information containing search statistics and principal variation</returns>
    ThinkInfo CreateThinkInfo(SearchResult searchResult)
    {
        Debug.Assert(_searcher is not null);

        return new ThinkInfo(searchResult.RootEval.PV.ToArray())
        {
            ElapsedMs = _searcher.SearchElapsedMs,
            NodeCount = _searcher.NodeCount,
            Nps = _searcher.Nps,
            Depth = searchResult.ChildEvals[0].PV.Length,
            EvalScore = searchResult.ChildEvals[0].ExpectedReward * 100.0,
        };
    }

    /// <summary>
    /// Creates multi-PV analysis results from search results, including all candidate moves.
    /// Filters out moves with invalid (NaN) expected rewards.
    /// </summary>
    /// <param name="searchResult">The search results to convert</param>
    /// <returns>Multi-PV analysis containing all valid candidate moves with their evaluations</returns>
    static MultiPV CreateMultiPV(SearchResult searchResult)
    {
        var multiPV = new MultiPV(searchResult.ChildEvals.Length);
        foreach (var childEval in searchResult.ChildEvals)
        {
            if (double.IsNaN(childEval.ExpectedReward))
                continue;

            multiPV.Add(new MultiPVItem(childEval.PV.ToArray())
            {
                Depth = childEval.PV.Length,
                NodeCount = childEval.SimulationCount,
                EvalScore = childEval.ExpectedReward * 100.0,
                EvalScoreType = EvalScoreType.WinRate
            });
        }
        return multiPV;
    }

    /// <summary>
    /// Converts search results to a formatted string for logging purposes.
    /// Includes search statistics and detailed information about each candidate move.
    /// </summary>
    /// <param name="searchResult">The search results to format</param>
    /// <returns>Formatted string representation of the search results</returns>
    string SearchResultToString(SearchResult searchResult)
    {
        Debug.Assert(_searcher is not null);

        var sb = new StringBuilder();
        sb.Append("elapsed=").Append(_searcher.SearchElapsedMs).Append("[ms] ");
        sb.Append(_searcher.NodeCount).Append("[nodes] ");
        sb.Append(_searcher.Nps).Append("[nps] ");
        sb.Append(searchResult.RootEval.SimulationCount).Append("[simulations] ");
        sb.Append("win_rate=").Append((searchResult.RootEval.ExpectedReward * 100.0).ToString("F2")).Append("%\n");
        sb.Append("|move|win_rate|effort|simulation|depth|pv\n");

        foreach (MoveEval childEval in searchResult.ChildEvals)
        {
            sb.Append("| ").Append(childEval.Move).Append(' ');
            sb.Append('|').Append((childEval.ExpectedReward * 100.0).ToString("F2").PadLeft(7));
            sb.Append('|').Append((childEval.Effort * 100.0).ToString("F2").PadLeft(5));
            sb.Append('|').Append(childEval.SimulationCount.ToString().PadLeft(10));
            sb.Append('|').Append(childEval.PV.Length.ToString().PadLeft(5));
            sb.Append('|');
            foreach (var move in childEval.PV)
                sb.Append(move).Append(' ');
            sb.Append('\n');
        }

        return sb.ToString();
    }

    /// <summary>
    /// Writes a message to the log file and optionally to the console if logging is enabled.
    /// Thread-safe through locking mechanism.
    /// </summary>
    /// <param name="msg">The message to write to the log</param>
    void WriteLog(string msg)
    {
        lock (_logger)
        {
            _logger.Write(msg);
            if (Options["show_log"])
                Console.Write(msg);
            _logger.Flush();
        }
    }

    /// <summary>
    /// Event handler called when the value function weights file path is changed.
    /// Reloads the value function and reinitializes the search tree if the engine is ready.
    /// </summary>
    /// <param name="option">The engine option containing the new file path</param>
    void OnValueFuncWeightsPathSpecified(EngineOption option)
    {
        string path = option;

        if (!File.Exists(path))
        {
            SendErrorMessage($"Cannot find a weights file of value function at \"{path}\".");
            return;
        }

        if (!IsReady)
            return;

        StopIfPondering();

        try
        {
            InitTree(ValueFunction.LoadFromFile(path));
        }
        catch (InvalidDataException ex)
        {
            SendErrorMessage(ex.Message);
        }
    }

    /// <summary>
    /// Event handler called when the number of search threads is changed.
    /// Updates the searcher's thread count after stopping any ongoing search.
    /// </summary>
    /// <param name="option">The engine option containing the new thread count</param>
    void OnNumThreadsChanged(EngineOption option)
    {
        if (_searcher is null)
        {
            SendErrorMessage("Specify weights file path of value function.");
            return;
        }

        StopIfPondering();

        _searcher.NumThreads = option;
    }

    /// <summary>
    /// Event handler called when the thought log file path is changed.
    /// Closes the current log file and opens a new one at the specified path.
    /// Cannot be changed while search is in progress.
    /// </summary>
    /// <param name="option">The engine option containing the new log file path</param>
    void OnThoughtLogPathChanged(EngineOption option)
    {
        if (_searcher is not null && _searcher.IsSearching)
        {
            SendErrorMessage("Cannot change thought log path while searching is in progress.");
            return;
        }

        _logger?.Dispose();
        _logger = new StreamWriter(Options["thought_log_path"].CurrentValue);
    }
}