global using MultiPV = System.Collections.Generic.List<ACG2025_reference_implementation.Engines.MultiPVItem>;
global using EngineOptions = System.Collections.Generic.Dictionary<string, ACG2025_reference_implementation.Engines.EngineOption>;

namespace ACG2025_reference_implementation.Engines;

using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

using ACG2025_reference_implementation.Reversi;

/// <summary>
/// Enumeration defining the types of evaluation scores used by engines.
/// </summary>
internal enum EvalScoreType
{
    /// <summary>Win rate evaluation (0.0 to 1.0)</summary>
    WinRate,
    /// <summary>Disc difference evaluation</summary>
    DiscDiff,
    /// <summary>Exact Win/Draw/Loss result</summary>
    ExactWDL,
    /// <summary>Exact disc difference at game end</summary>
    ExactDiscDiff,
    /// <summary>Other evaluation type</summary>
    Other
}

/// <summary>
/// Class containing thinking information reported by engines during search.
/// </summary>
/// <param name="pv">Principal variation (sequence of best moves)</param>
internal class ThinkInfo(IEnumerable<BoardCoordinate> pv)
{
    /// <summary>Elapsed time in milliseconds</summary>
    public int? ElapsedMs { get; init; }
    /// <summary>Number of nodes searched</summary>
    public long? NodeCount { get; init; }
    /// <summary>Nodes per second</summary>
    public double? Nps { get; init; }
    /// <summary>Search depth</summary>
    public int? Depth { get; init; }
    /// <summary>Evaluation score</summary>
    public double? EvalScore { get; init; }
    /// <summary>Principal variation (best line of play)</summary>
    public ReadOnlySpan<BoardCoordinate> PrincipalVariation => _pv;

    readonly BoardCoordinate[] _pv = [.. pv];
}

/// <summary>
/// Class representing a single item in Multi-PV analysis (multiple principal variations).
/// Used to show multiple candidate moves with their evaluations.
/// </summary>
/// <param name="pv">Principal variation for this candidate move</param>
internal class MultiPVItem(IEnumerable<BoardCoordinate> pv)
{
    /// <summary>Search depth for this variation</summary>
    public int? Depth { get; init; }
    /// <summary>Number of nodes searched for this variation</summary>
    public long? NodeCount { get; init; }
    /// <summary>Evaluation score for this variation</summary>
    public double? EvalScore { get; init; }
    /// <summary>Type of evaluation score</summary>
    public EvalScoreType EvalScoreType { get; init; } = EvalScoreType.Other;
    /// <summary>Exact game result (Win/Draw/Loss) if known</summary>
    public GameResult ExactWDL { get; init; } = GameResult.NotOver;
    /// <summary>Exact disc difference if game result is known</summary>
    public int? ExactDiscDiff { get; init; }
    /// <summary>Principal variation (sequence of moves) for this candidate</summary>
    public ReadOnlySpan<BoardCoordinate> PrincipalVariation => _pv;

    BoardCoordinate[] _pv = [.. pv];
}

/// <summary>
/// Class representing a move decided by an engine, including evaluation information.
/// </summary>
internal class EngineMove
{
    /// <summary>Board coordinate of the move</summary>
    public BoardCoordinate Coord { get; set; } = BoardCoordinate.Null;
    /// <summary>Evaluation score for this move</summary>
    public double? EvalScore { get; set; }
    /// <summary>Type of evaluation score</summary>
    public EvalScoreType EvalScoreType { get; set; }
    /// <summary>Time elapsed to find this move in milliseconds</summary>
    public int? ElapsedMs { get; set; }

    /// <summary>
    /// Initializes a new instance of EngineMove with default values.
    /// </summary>
    public EngineMove() { }

    /// <summary>
    /// Initializes a new instance of EngineMove with the specified coordinate.
    /// </summary>
    /// <param name="coord">Board coordinate of the move</param>
    public EngineMove(BoardCoordinate coord) : this(coord, null, EvalScoreType.Other, null) { }

    /// <summary>
    /// Initializes a new instance of EngineMove with complete information.
    /// </summary>
    /// <param name="coord">Board coordinate of the move</param>
    /// <param name="evalScore">Evaluation score</param>
    /// <param name="evalScoreType">Type of evaluation score</param>
    /// <param name="elapsedMs">Time elapsed in milliseconds</param>
    public EngineMove(BoardCoordinate coord, double? evalScore, EvalScoreType evalScoreType, int? elapsedMs)
    {
        Coord = coord;
        EvalScore = evalScore;
        EvalScoreType = evalScoreType;
        ElapsedMs = elapsedMs;
    }
}

internal class EngineOption
{
    public string DefaultValue { get; }
    public long MinValue { get; } = long.MinValue;
    public long MaxValue { get; } = long.MaxValue;
    public bool IsInteger { get; }
    public event Action<EngineOption> ValueChanged = _ => { };

    string _currentValue;

    public EngineOption(bool value)
    {
        DefaultValue = _currentValue = value.ToString();
        IsInteger = false;
    }

    public EngineOption(string value)
    {
        DefaultValue = _currentValue = value;
        IsInteger = false;
    }

    public EngineOption(long value, long min = long.MinValue, long max = long.MaxValue)
    {
        DefaultValue = _currentValue = value.ToString();
        IsInteger = true;
        (MinValue, MaxValue) = (min, max);
    }

    public string CurrentValue
    {
        get => _currentValue;

        set
        {
            if (IsInteger)
            {
                var v = long.Parse(value);
                if (MinValue <= v && v <= MaxValue)
                    _currentValue = value;
                else
                    throw new ArgumentOutOfRangeException(nameof(value));

                ValueChanged(this);
                return;
            }

            _currentValue = value;
            ValueChanged(this);
            return;
        }
    }

    public static implicit operator int(EngineOption option)
    {
        if (option.IsInteger)
            return int.Parse(option.CurrentValue);
        throw new InvalidCastException("Cannot cast non-integer EngineOption to long");
    }

    public static implicit operator long(EngineOption option)
    {
        if (option.IsInteger)
            return long.Parse(option.CurrentValue);
        throw new InvalidCastException("Cannot cast non-integer EngineOption to long");
    }

    public static implicit operator string(EngineOption option) => option.CurrentValue;

    public static implicit operator bool(EngineOption option)
    {
        if (bool.TryParse(option.CurrentValue, out var flag))
            return flag;
        throw new InvalidCastException("Cannot cast non-boolean EngineOption to bool");
    }
}

/// <summary>
/// Abstract base class for Reversi game engines.
/// Provides common functionality for position management, move handling, and communication with protocols.
/// </summary>
/// <param name="name">Name of the engine</param>
/// <param name="version">Version of the engine</param>
/// <param name="author">Author of the engine</param>
internal abstract class Engine(string name, string version, string author)
{
    /// <summary>Name of the engine</summary>
    public string Name { get; private set; } = name;
    /// <summary>Version of the engine</summary>
    public string Version { get; private set; } = version;
    /// <summary>Author of the engine</summary>
    public string Author { get; private set; } = author;

    public bool IsReady { get; private set; } = false;

    /// <summary>Current side to move</summary>
    public DiscColor SideToMove => _position.SideToMove;

    /// <summary>Type of evaluation score this engine uses</summary>
    public EvalScoreType EvalScoreType { get; protected set; } = EvalScoreType.Other;

    /// <summary>Minimum evaluation score value</summary>
    public double EvalScoreMin { get; protected set; } = 0.0f;
    /// <summary>Maximum evaluation score value</summary>
    public double EvalScoreMax { get; protected set; } = 0.0f;

    /// <summary>Event fired when the engine sends a text message</summary>
    public event EventHandler<string> MessageWasSent = delegate { };
    /// <summary>Event fired when the engine sends an error message</summary>
    public event EventHandler<string> ErrorMessageWasSent = delegate { };
    /// <summary>Event fired when the engine sends thinking information</summary>
    public event EventHandler<ThinkInfo> ThinkInfoWasSent = delegate { };
    /// <summary>Event fired when the engine sends multi-PV analysis results</summary>
    public event EventHandler<MultiPV> MultiPVWereSent = delegate { };
    /// <summary>Event fired when the engine makes a move</summary>
    public event EventHandler<EngineMove> MoveWasSent = delegate { };
    /// <summary>Event fired when analysis is completed</summary>
    public event EventHandler AnalysisEnded = delegate { };

    /// <summary>Current position on the board</summary>
    protected Position Position => _position;
    /// <summary>Read-only collection of move history</summary>
    protected ReadOnlyCollection<Move> MoveHistory => new(_moveHistory);
    /// <summary>Engine-specific configuration options</summary>
    protected EngineOptions Options = [];

    Position _position = new();
    readonly List<Move> _moveHistory = [];

    /// <summary>
    /// Gets a copy of the current position.
    /// </summary>
    /// <returns>Copy of the current position</returns>
    public Position GetPosition() => _position;

    /// <summary>
    /// Initializes the engine and prepares it for use.
    /// </summary>
    public bool Ready() => IsReady = OnReady();

    /// <summary>
    /// Initializes the position to the specified state and clears move history.
    /// </summary>
    /// <param name="pos">Position to initialize to</param>
    public void InitPosition(ref Position pos)
    {
        _position = pos;
        _moveHistory.Clear();
        OnInitializedPosition();
    }

    /// <summary>
    /// Updates the position with the specified move, handling color changes and passes automatically.
    /// </summary>
    /// <param name="color">Color of the player making the move</param>
    /// <param name="moveCoord">Coordinate of the move to make</param>
    /// <returns>True if the move was successfully applied, false if illegal</returns>
    public bool UpdatePosition(DiscColor color, BoardCoordinate moveCoord)
    {
        if (color != _position.SideToMove)
        {
            _position.Pass();
            _moveHistory.Add(Move.Pass);
        }

        if (moveCoord == BoardCoordinate.Pass)
        {
            _position.Pass();
            _moveHistory.Add(Move.Pass);
            OnUpdatedPosition(Move.Pass);
            return true;
        }

        if (!_position.IsLegalMoveAt(moveCoord))
            return false;

        var move = _position.CalcFlip(moveCoord);
        _position.Update(ref move);
        _moveHistory.Add(move);
        OnUpdatedPosition(move);
        return true;
    }

    /// <summary>
    /// Undoes the last move and removes it from the move history.
    /// </summary>
    /// <returns>True if a move was successfully undone, false if no moves to undo</returns>
    public bool UndoPosition()
    {
        if (_moveHistory.Count == 0)
            return false;

        var move = _moveHistory.Last();
        _position.Undo(ref move);
        _moveHistory.RemoveAt(_moveHistory.Count - 1);
        OnUndonePosition(move);
        return true;
    }

    /// <summary>
    /// Sets the value of a specific engine configuration option.
    /// </summary>
    /// <param name="name">The name of the option to set</param>
    /// <param name="value">The new value for the option as a string</param>
    /// <returns>True if the option was found and set successfully, false if the option does not exist</returns>
    public bool SetOption(string name, string value)
    {
        if (!Options.TryGetValue(name, out EngineOption? option))
            return false;

        try
        {
            option.CurrentValue = value;
        }
        catch (Exception ex) when (ex is ArgumentOutOfRangeException || ex is OverflowException)
        {
            if (option.IsInteger)
            {
                SendErrorMessage($"Value '{value}' is out of range. Must be between {option.MinValue} and {option.MaxValue}.");
                return false;
            }

            throw;
        }
        catch (FormatException)
        {
            if (option.IsInteger)
            {
                SendErrorMessage($"Value '{value}' is not a valid integer.");
                return false;
            }

            throw;
        }

        return true;
    }

    /// <summary>Terminates the engine and cleans up resources</summary>
    public abstract void Quit();

    /// <summary>
    /// Sets the main thinking time for the specified player color.
    /// This is the initial time available before entering byoyomi phase.
    /// </summary>
    /// <param name="color">The player color to set main time for</param>
    /// <param name="mainTimeMs">Main time in milliseconds</param>
    public abstract void SetMainTime(DiscColor color, int mainTimeMs);

    /// <summary>
    /// Sets the byoyomi time for the specified player color.
    /// This is the time limit for making moves after main time is exhausted.
    /// </summary>
    /// <param name="color">The player color to set byoyomi time for</param>
    /// <param name="byoyomiMs">Byoyomi time in milliseconds</param>
    public abstract void SetByoyomi(DiscColor color, int byoyomiMs);

    /// <summary>
    /// Sets the number of stones (moves) that must be played within byoyomi time.
    /// Used in Canadian time system where multiple moves must be made in the byoyomi period.
    /// </summary>
    /// <param name="color">The player color to set byoyomi stones for</param>
    /// <param name="byoyomiStones">Number of moves to be made within byoyomi time</param>
    public abstract void SetByoyomiStones(DiscColor color, int byoyomiStones);

    /// <summary>
    /// Sets the time increment added after each move (Fischer time system).
    /// Time is added to the player's clock after making each move.
    /// </summary>
    /// <param name="color">The player color to set time increment for</param>
    /// <param name="incMs">Time increment in milliseconds</param>
    public abstract void SetTimeIncrement(DiscColor color, int incMs);

    /// <summary>Sets the engine's thinking level or strength</summary>
    /// <param name="level">Level to set</param>
    public abstract void SetLevel(int level);

    /// <summary>Starts thinking and makes a move</summary>
    public abstract void Go();

    /// <summary>Analyzes the position and provides multiple candidate moves</summary>
    /// <param name="numMoves">Number of candidate moves to analyze</param>
    public abstract void Analyze(int numMoves);

    /// <summary>Stops the engine's thinking process</summary>
    /// <param name="timeoutMs">Maximum time to wait for stop in milliseconds</param>
    /// <returns>True if thinking was stopped successfully</returns>
    public abstract bool StopThinking(int timeoutMs);

    /// <summary>Sends a text message to listeners</summary>
    /// <param name="msg">Message to send</param>
    protected void SendTextMessage(string msg) => MessageWasSent(this, msg);
    /// <summary>Sends an error message to listeners</summary>
    /// <param name="errMsg">Error message to send</param>
    protected void SendErrorMessage(string errMsg) => ErrorMessageWasSent(this, errMsg);
    /// <summary>Sends thinking information to listeners</summary>
    /// <param name="thinkInfo">Thinking information to send</param>
    protected void SendThinkInfo(ThinkInfo thinkInfo) => ThinkInfoWasSent(this, thinkInfo);
    /// <summary>Sends multi-PV analysis results to listeners</summary>
    /// <param name="multiPV">Multi-PV results to send</param>
    protected void SendMultiPV(MultiPV multiPV) => MultiPVWereSent(this, multiPV);
    /// <summary>Sends the decided move to listeners</summary>
    /// <param name="move">Move to send</param>
    protected void SendMove(EngineMove move) => MoveWasSent(this, move);
    protected void EndAnalysis() => AnalysisEnded(this, EventArgs.Empty);

    /// <summary>Called when the engine should initialize and prepare for use</summary>
    /// <returns>True if initialization succeeded, false otherwise</returns>
    protected abstract bool OnReady();
    /// <summary>Called when the initial position is set</summary>
    protected abstract void OnInitializedPosition();
    /// <summary>Called when a move is made on the board</summary>
    /// <param name="move">The move that was made</param>
    protected abstract void OnUpdatedPosition(Move move);
    /// <summary>Called when a move is undone</summary>
    /// <param name="move">The move that was undone</param>
    protected abstract void OnUndonePosition(Move move);
}
