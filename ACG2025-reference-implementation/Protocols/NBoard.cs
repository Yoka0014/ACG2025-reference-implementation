namespace ACG2025_reference_implementation.Protocols;

using System;
using System.IO;
using System.Text;
using System.Threading;
using System.Collections.Generic;

using ACG2025_reference_implementation.Utils;
using ACG2025_reference_implementation.GameFormats;
using ACG2025_reference_implementation.Reversi;
using ACG2025_reference_implementation.Engines;

using CommandHandler = System.Action<Utils.Tokenizer>;

/// <summary>
/// Implementation of the NBoard protocol for communication with game engines.
/// This class provides a command-line interface that supports various NBoard protocol commands
/// for controlling game engines, including game setup, move execution, and analysis.
/// </summary>
internal class NBoard : IProtocol
{
    /// <summary>
    /// The supported NBoard protocol version.
    /// </summary>
    public const int ProtocolVersion = 2;

    /// <summary>
    /// The timeout duration in milliseconds for stopping engine operations.
    /// </summary>
    const int TimeoutMs = 10000;

    /// <summary>
    /// Input stream for reading commands.
    /// </summary>
    readonly TextReader _cmdIn;
    
    /// <summary>
    /// Output streams for command responses and error messages.
    /// </summary>
    readonly TextWriter _cmdOut, _errOut;
    
    /// <summary>
    /// Dictionary mapping command names to their corresponding handler methods.
    /// </summary>
    readonly Dictionary<string, CommandHandler> _commands = [];

    /// <summary>
    /// The game engine instance being controlled by this protocol implementation.
    /// </summary>
    Engine? _engine;
    
    /// <summary>
    /// Logger for recording protocol communication.
    /// </summary>
    StreamWriter _logger;
    
    /// <summary>
    /// The number of hints to provide during analysis.
    /// </summary>
    volatile int _numHints = 1;
    
    /// <summary>
    /// Flag indicating whether the engine is currently thinking or analyzing.
    /// </summary>
    volatile bool _engineIsThinking;
    
    /// <summary>
    /// Flag indicating whether the protocol should quit.
    /// </summary>
    volatile bool _quitFlag;
    
    /// <summary>
    /// Lock object for synchronizing mainloop execution.
    /// </summary>
    readonly object _mainloopLock = new();

    /// <summary>
    /// Initializes a new instance of the NBoard class using standard console streams.
    /// </summary>
    public NBoard() : this(Console.In, Console.Out, Console.Error) { }

    /// <summary>
    /// Initializes a new instance of the NBoard class with custom input and output streams.
    /// </summary>
    /// <param name="cmdIn">The input stream for reading commands.</param>
    /// <param name="cmdOut">The output stream for sending responses.</param>
    /// <param name="errOut">The output stream for sending error messages.</param>
    public NBoard(TextReader cmdIn, TextWriter cmdOut, TextWriter errOut)
    {
        _cmdIn = cmdIn;
        _cmdOut = cmdOut;
        _errOut = errOut;
        _logger = new StreamWriter(Stream.Null);
        InitCommandHandlers();
    }

    /// <summary>
    /// Initializes the command handlers dictionary with all supported NBoard protocol commands.
    /// </summary>
    void InitCommandHandlers()
    {
        _commands["nboard"] = ExecuteNboardCommand;
        _commands["set"] = ExecuteSetCommand;
        _commands["move"] = ExecuteMoveCommand;
        _commands["hint"] = ExecuteHintCommand;
        _commands["go"] = ExecuteGoCommand;
        _commands["ping"] = ExecutePingCommand;
        _commands["learn"] = ExecuteLearnCommand;
        _commands["analyze"] = ExecuteAnalyzeCommand;
        _commands["quit"] = ExecuteQuitCommand;

        // Set command sub-handlers
        _commands["depth"] = ExecuteSetDepthCommand;
        _commands["game"] = ExecuteSetGameCommand;
        _commands["contempt"] = ExecuteSetContemptCommand;
        _commands["time"] = ExecuteSetTimeCommand;  // Extension to standard NBoard protocol
    }

    /// <summary>
    /// Starts the main protocol loop with the specified engine.
    /// </summary>
    /// <param name="engine">The game engine to control.</param>
    public void Mainloop(Engine? engine) => Mainloop(engine, null);

    /// <summary>
    /// Starts the main protocol loop with the specified engine and optional logging.
    /// </summary>
    /// <param name="engine">The game engine to control.</param>
    /// <param name="logFilePath">Optional path to a log file for recording protocol communication.</param>
    public void Mainloop(Engine? engine, string? logFilePath)
    {
        if (engine is null)
            throw new ArgumentNullException(nameof(engine), "Specified engine was null.");

        try
        {
            if (!Monitor.TryEnter(_mainloopLock))
                throw new InvalidOperationException("Cannot execute multiple mainloop.");

            InitEngine(engine);
            _quitFlag = false;
            _logger = (logFilePath is not null) ? new StreamWriter(logFilePath) : new StreamWriter(Stream.Null);

            string cmdName;
            string? line;
            while (!_quitFlag)
            {
                line = _cmdIn.ReadLine();
                if (line is null)
                    break;

                _logger.WriteLine($"< {line}\n");

                var tokenizer = new Tokenizer { Input = line };
                cmdName = tokenizer.ReadNext();
                if (!_commands.TryGetValue(cmdName, out CommandHandler? handler))
                    Fail($"Unknown command: {cmdName}");
                else
                    handler(tokenizer);
            }
        }
        finally
        {
            if (Monitor.IsEntered(_mainloopLock))
                Monitor.Exit(_mainloopLock);
        }
    }

    /// <summary>
    /// Initializes the engine and sets up event handlers for engine responses.
    /// </summary>
    /// <param name="engine">The game engine to initialize.</param>
    void InitEngine(Engine engine)
    {
        _engine = engine;
        engine.ErrorMessageWasSent += (sender, msg) => Fail(msg);
        engine.ThinkInfoWasSent += (sender, thinkInfo) => SendNodeStats(thinkInfo);
        engine.MultiPVWereSent += (sender, multiPV) => SendHints(multiPV);
        engine.MoveWasSent += (sender, move) => SendMove(move);
        engine.AnalysisEnded += (sender, _) => { Succeed("status"); _engineIsThinking = false; };
    }

    /// <summary>
    /// Sends a successful response message to the output stream and logs it.
    /// </summary>
    /// <param name="responce">The response message to send.</param>
    void Succeed(string responce)
    {
        _cmdOut.WriteLine(responce);
        _cmdOut.Flush();

        _logger.WriteLine($"> {responce}");
        _logger.Flush();
    }

    /// <summary>
    /// Sends an error message to the error stream and logs it.
    /// </summary>
    /// <param name="message">The error message to send.</param>
    void Fail(string message)
    {
        _errOut.WriteLine($"Error: {message}");
        _errOut.Flush();

        _logger.WriteLine($">! {message}");
        _logger.Flush();
    }

    /// <summary>
    /// Sends node statistics information from the engine's thinking process.
    /// </summary>
    /// <param name="thinkInfo">The thinking information containing node count and elapsed time.</param>
    void SendNodeStats(ThinkInfo thinkInfo)
    {
        if (!thinkInfo.NodeCount.HasValue)
            return;

        var sb = new StringBuilder();
        sb.Append("nodestats ").Append(thinkInfo.NodeCount.Value).Append(' ');

        if (thinkInfo.ElapsedMs.HasValue)
            sb.Append(thinkInfo.ElapsedMs.Value * 1.0e-3);

        Succeed(sb.ToString());
    }

    /// <summary>
    /// Sends hint information containing principal variations and evaluation scores.
    /// </summary>
    /// <param name="multiPV">The multi-PV data containing multiple principal variations.</param>
    void SendHints(MultiPV multiPV)
    {
        var sb = new StringBuilder();
        for (var i = 0; i < _numHints && i < multiPV.Count; i++)
        {
            var pv = multiPV[i];

            sb.Append("search ");
            foreach (var move in pv.PrincipalVariation)
            {
                if (move != BoardCoordinate.Pass)
                    sb.Append(move);
                else
                    sb.Append("PA");
            }

            if (pv.EvalScore.HasValue)
            {
                if (pv.EvalScoreType != EvalScoreType.WinRate)
                    sb.Append($" {pv.EvalScore.Value:f2}");
                else
                    sb.Append($" {pv.EvalScore.Value * 100.0 - 50.0:f2}");
            }
            else
                sb.Append(' ').Append(0);

            sb.Append(" 0 ");

            if (pv.EvalScoreType == EvalScoreType.ExactWDL)
                sb.Append("100%W");
            else if (pv.EvalScoreType == EvalScoreType.ExactDiscDiff)
                sb.Append("100%");
            else
                sb.Append(pv.Depth ?? 0);

            sb.Append('\n');
        }

        Succeed(sb.ToString());
    }

    /// <summary>
    /// Sends the best move determined by the engine along with evaluation and timing information.
    /// </summary>
    /// <param name="move">The engine move containing coordinate, evaluation, and elapsed time.</param>
    void SendMove(EngineMove move)
    {
        var sb = new StringBuilder();
        sb.Append("=== ");

        if (move.Coord == BoardCoordinate.Pass)
            sb.Append("PA");
        else
            sb.Append(move.Coord);

        if (move.EvalScore.HasValue)
        {
            if (move.EvalScoreType != EvalScoreType.WinRate)
                sb.Append('/').Append(move.EvalScore.Value);
            else
                sb.Append('/').Append(move.EvalScore.Value * 100.0 - 50.0);
        }

        if (move.ElapsedMs.HasValue)
            sb.Append('/').Append(move.ElapsedMs.Value);

        Succeed(sb.ToString());
        Succeed("status");
    }

    /// <summary>
    /// Executes the 'nboard' command to establish protocol version compatibility.
    /// </summary>
    /// <param name="tokenizer">The tokenizer containing command arguments.</param>
    void ExecuteNboardCommand(Tokenizer tokenizer)
    {
        var token = tokenizer.ReadNext();
        if (!int.TryParse(token, out int version))
        {
            Fail("NBoard version must be an integer.");
            return;
        }

        if (version != ProtocolVersion)
        {
            Fail($"NBoard version {version} is not supported.");
            return;
        }

        if (_engine!.Ready())
            Succeed($"set myname {_engine?.Name}");
    }

    /// <summary>
    /// Executes the 'set' command to configure various engine properties.
    /// </summary>
    /// <param name="tokenizer">The tokenizer containing command arguments.</param>
    void ExecuteSetCommand(Tokenizer tokenizer)
    {
        var propertyName = tokenizer.ReadNext();

        if (!_commands.TryGetValue(propertyName, out CommandHandler? handler))
        {
            Fail($"Unknown property: {propertyName}");
            return;
        }

        handler(tokenizer);
    }

    /// <summary>
    /// Executes the 'set depth' command to configure the engine's search depth.
    /// </summary>
    /// <param name="tokenizer">The tokenizer containing command arguments.</param>
    void ExecuteSetDepthCommand(Tokenizer tokenizer)
    {
        var token = tokenizer.ReadNext();
        if (!int.TryParse(token, out int depth))
        {
            Fail("Depth must be an integer.");
            return;
        }

        if (depth < 1 || depth > 60)
        {
            Fail("Depth must be within [1, 60].");
            return;
        }

        _engine?.SetLevel(depth);
    }

    /// <summary>
    /// Executes the 'set game' command to initialize the game position from a GGF string.
    /// </summary>
    /// <param name="tokenizer">The tokenizer containing the GGF game data.</param>
    void ExecuteSetGameCommand(Tokenizer tokenizer)
    {
        if (_engineIsThinking && !_engine!.StopThinking(TimeoutMs))
        {
            Fail("Cannot suspend current thinking task.");
            return;
        }

        GGFReversiGame game;
        try
        {
            game = new GGFReversiGame(tokenizer.ReadToEnd());
        }
        catch (GGFParserException ex)
        {
            Fail($"Cannot parse GGF string. \nDetail: {ex.Message}\nStack trace:\n\t{ex.StackTrace}");
            return;
        }

        var pos = game.GetPosition();
        foreach (GGFMove move in game.Moves)
        {
            if (!pos.Update(move.Coord))
            {
                Fail($"Specified moves contain an invalid move {move.Coord}.");
                return;
            }
        }

        var currentPos = _engine!.GetPosition();
        Span<Move> moves = stackalloc Move[Constants.MaxLegalMoves];
        var num_moves = currentPos.GetNextMoves(ref moves);
        var updated = false;
        for (var i = 0; i < num_moves; i++)
        {
            ref var move = ref moves[i];
            currentPos.CalcFlip(ref move);
            currentPos.Update(ref move);

            if (currentPos == pos)
            {
                _engine.UpdatePosition(currentPos.OpponentColor, move.Coord);
                updated = true;
                break;
            }

            currentPos.Undo(ref move);
        }

        if (!updated)
            _engine.InitPosition(ref pos);

        var times = new TimeControl[] { game.BlackThinkingTime, game.WhiteThinkingTime };
        for (var color = DiscColor.Black; color <= DiscColor.White; color++)
        {
            var time = times[(int)color];
            if (time.MainTimeMs > 0)
            {
                _engine.SetMainTime(color, time.MainTimeMs);
                _engine.SetTimeIncrement(color, time.IncrementMs);
            }
        }
    }

    /// <summary>
    /// Executes the 'set contempt' command. This feature is not supported.
    /// </summary>
    /// <param name="tokenizer">The tokenizer containing command arguments.</param>
    void ExecuteSetContemptCommand(Tokenizer tokenizer) => Fail("Contempt is not supported.");

    /// <summary>
    /// Executes the 'set time' command to configure time control settings for a specific color.
    /// Supports main time, increment, and byoyomi settings.
    /// </summary>
    /// <param name="tokenizer">The tokenizer containing time control parameters.</param>
    void ExecuteSetTimeCommand(Tokenizer tokenizer)
    {
        // Format: set time [color] main [int] inc [int] byoyomi [int]
        // The main, inc, and byoyomi parameters can be omitted.

        var token = tokenizer.ReadNext().ToLower();
        DiscColor color;
        if (token == "b" || token == "black")
            color = DiscColor.Black;
        else if (token == "w" || token == "white")
            color = DiscColor.White;
        else
        {
            Fail("Specify a valid color.");
            return;
        }

        string timeStr;
        int timeMs;
        while (!tokenizer.IsEndOfString)
        {
            token = tokenizer.ReadNext().ToLower();
            timeStr = tokenizer.ReadNext().ToLower();

            switch (token)
            {
                case "time":
                    if (!tryParseTime(timeStr, out timeMs))
                        return;
                    _engine?.SetMainTime(color, timeMs);
                    break;

                case "inc":
                    if (!tryParseTime(timeStr, out timeMs))
                        return;
                    _engine?.SetTimeIncrement(color, timeMs);
                    break;

                case "byoyomi":
                    if (!tryParseTime(timeStr, out timeMs))
                        return;
                    _engine?.SetByoyomi(color, timeMs);
                    break;

                default:
                    Fail($"\"{token}\" is an invalid token.");
                    return;
            }
        }

        bool tryParseTime(string str, out int timeMs)
        {
            if (!int.TryParse(str, out timeMs))
            {
                Fail($"Time must be an integer.");
                return false;
            }
            return true;
        }
    }

    /// <summary>
    /// Executes the 'move' command to update the game position with a player's move.
    /// </summary>
    /// <param name="tokenizer">The tokenizer containing the move coordinate.</param>
    void ExecuteMoveCommand(Tokenizer tokenizer)
    {
        if (_engineIsThinking && !_engine!.StopThinking(TimeoutMs))
        {
            Fail("Cannot suspend current thinking task.");
            return;
        }

        var moveStr = tokenizer.ReadTo('/').Trim().ToLower();
        var move = (moveStr == "pa") ? BoardCoordinate.Pass : Reversi.Utils.ParseCoordinate(moveStr);

        if (move == BoardCoordinate.Null)
        {
            Fail($"Specify a valid move coordinate.");
            return;
        }

        if (!_engine!.UpdatePosition(_engine.SideToMove, move))
        {
            Fail($"Move {move} is invalid.");
            return;
        }

        // Note: Evaluation score and time information are ignored.
    }

    /// <summary>
    /// Executes the 'hint' command to start engine analysis and provide move suggestions.
    /// </summary>
    /// <param name="tokenizer">The tokenizer containing the number of hints to provide.</param>
    void ExecuteHintCommand(Tokenizer tokenizer)
    {
        if (!int.TryParse(tokenizer.ReadNext(), out var numHints))
        {
            Fail("The number of hints must be an integer.");
            return;
        }

        if (numHints < 1)
        {
            Fail("The number of hints must be more than or equal 1.");
            return;
        }

        _numHints = numHints;

        Succeed("status Analysing");
        _engineIsThinking = true;
        _engine?.Analyze(_numHints);
    }

    /// <summary>
    /// Executes the 'go' command to start the engine thinking for the best move.
    /// </summary>
    /// <param name="tokenizer">The tokenizer containing command arguments.</param>
    void ExecuteGoCommand(Tokenizer tokenizer)
    {
        Succeed("status Thinking");
        _engineIsThinking = true;
        _engine?.Go();
    }

    /// <summary>
    /// Executes the 'ping' command to test protocol responsiveness.
    /// </summary>
    /// <param name="tokenizer">The tokenizer containing an optional ping identifier.</param>
    void ExecutePingCommand(Tokenizer tokenizer)
    {
        if (_engineIsThinking && !_engine!.StopThinking(TimeoutMs))
        {
            Fail("Cannot suspend current thinking task.");
            return;
        }

        if (!int.TryParse(tokenizer.ReadNext(), out var n))
            n = 0;

        Succeed($"pong {n}");
    }

    /// <summary>
    /// Executes the 'learn' command. This feature is not supported.
    /// </summary>
    /// <param name="tokenizer">The tokenizer containing command arguments.</param>
    void ExecuteLearnCommand(Tokenizer tokenizer) => Fail("learn command is not supported.");

    /// <summary>
    /// Executes the 'analyze' command. This feature is not supported.
    /// </summary>
    /// <param name="tokenizer">The tokenizer containing command arguments.</param>
    void ExecuteAnalyzeCommand(Tokenizer tokenizer) => Fail("Not supported.");

    /// <summary>
    /// Executes the 'quit' command to terminate the protocol session gracefully.
    /// </summary>
    /// <param name="tokenizer">The tokenizer containing command arguments.</param>
    void ExecuteQuitCommand(Tokenizer tokenizer)
    {
        if (_quitFlag)
            return;

        ExecutePingCommand(new Tokenizer());
        _quitFlag = true;
    }
}
