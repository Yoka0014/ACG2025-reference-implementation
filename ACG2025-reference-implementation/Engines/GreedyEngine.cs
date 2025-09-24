namespace ACG2025_reference_implementation.Engines;

using System;
using System.IO;
using System.Linq;

using ACG2025_reference_implementation.Reversi;
using ACG2025_reference_implementation.Evaluation;
using ACG2025_reference_implementation.Search;

/// <summary>
/// A greedy engine class for Reversi.
/// Selects the move with the highest value function score using one-ply lookahead.
/// </summary>
/// <param name="valueFuncWeightsPath">Path to the value function weights file</param>
internal class GreedyEngine(string valueFuncWeightsPath) : Engine("GreedyEngine", "0.0", "Yoka0014")
{
    /// <summary>
    /// Default value function weights file path
    /// </summary>
    const string DefaultValueFuncWeightsPath = "params/value_func_weights.bin";

    /// <summary>
    /// Path to the value function weights file
    /// </summary>
    readonly string _valueFuncWeightsPath = valueFuncWeightsPath;
    
    /// <summary>
    /// Game state management object containing position and feature vector
    /// </summary>
    State _state;
    
    /// <summary>
    /// Value function object for win rate prediction
    /// </summary>
    ValueFunction? _valueFunc;

    /// <summary>
    /// Initializes a GreedyEngine instance with default parameters.
    /// </summary>
    public GreedyEngine() : this(DefaultValueFuncWeightsPath) { }

    /// <summary>
    /// Terminates the engine.
    /// </summary>
    public override void Quit() { }
    /// <summary>
    /// Sets the engine's thinking level.
    /// Level setting is not used in this greedy engine.
    /// </summary>
    /// <param name="level">Level to set (unused)</param>
    public override void SetLevel(int level) { }

    /// <summary>
    /// Calculates the best move for the current board position and makes a decision.
    /// Evaluates all legal moves and selects the one with the highest win rate.
    /// </summary>
    public override void Go()
    {
        if (_valueFunc is null)
        {
            SendErrorMessage("Engine is not initialized.");
            return;
        }

        var multiPV = CreateMultiPV();

        if (multiPV is null)
        {
            SendMove(new EngineMove(BoardCoordinate.Pass));
            return;
        }

        var maxItem = multiPV.MaxBy(x => x.EvalScore);
        SendMove(new EngineMove
        {
            Coord = maxItem!.PrincipalVariation[0],
            EvalScore = maxItem.EvalScore,
            EvalScoreType = maxItem.EvalScoreType
        });
    }

    /// <summary>
    /// Analyzes the specified number of hints (candidate moves) and returns them sorted by evaluation score.
    /// Evaluates all legal moves and sends a MultiPV sorted by highest evaluation scores.
    /// </summary>
    /// <param name="numHints">Number of hints to return</param>
    public override void Analyze(int numHints)
    {
        if (_valueFunc is null)
        {
            SendErrorMessage("Engine is not initialized.");
            return;
        }

        var multiPV = CreateMultiPV();
        if(multiPV is not null)
            SendMultiPV(multiPV.OrderByDescending(x => x.EvalScore).ToList()[..Math.Min(numHints, multiPV.Count)]);
    }

    /// <summary>
    /// Stops thinking.
    /// The greedy engine makes decisions immediately, so always returns true.
    /// </summary>
    /// <param name="timeoutMs">Timeout in milliseconds (unused)</param>
    /// <returns>Always true</returns>
    public override bool StopThinking(int timeoutMs) => true;

    /// <summary>
    /// Performs engine initialization.
    /// Loads the value function weights file and initializes the state management object.
    /// </summary>
    /// <returns>True if initialization succeeds, false if it fails</returns>
    protected override bool OnReady()
    {
        if (!File.Exists(_valueFuncWeightsPath))
        {
            SendErrorMessage($"Cannot find value function's weights file at \"{_valueFuncWeightsPath}\"");
            return false;
        }

        try
        {
            _valueFunc = ValueFunction.LoadFromFile(_valueFuncWeightsPath);
        }
        catch (InvalidDataException ex)
        {
            SendErrorMessage(ex.Message);
            return false;
        }

        _state = new State(_valueFunc.NTupleManager);
        return true;
    }

    /// <summary>
    /// Performs processing when the initial board position is set.
    /// Initializes the state management object with the current board position.
    /// </summary>
    protected override void OnInitializedPosition()
    {
        var pos = Position;
        _state.Init(ref pos);
    }

    /// <summary>
    /// Performs state update processing when a move is made.
    /// </summary>
    /// <param name="move">Information about the move that was made</param>
    protected override void OnUpdatedPosition(Move move) => _state.Update(ref move);

    /// <summary>
    /// Performs state restoration processing when a move is undone.
    /// </summary>
    /// <param name="move">Information about the move to undo</param>
    protected override void OnUndonePosition(Move move) => _state.Undo(ref move);

    /// <summary>
    /// Evaluates all legal moves and creates a MultiPV.
    /// Performs one-ply lookahead for each move, evaluates with the value function, and generates a candidate move list.
    /// </summary>
    /// <returns>List of candidate moves, or null if no legal moves exist</returns>
    MultiPV? CreateMultiPV()
    {
        Span<Move> moves = stackalloc Move[Constants.MaxLegalMoves];
        var numMoves = _state.Position.GetNextMoves(ref moves);

        if (numMoves == 0)
            return null;

        Span<double> values = stackalloc double[numMoves];
        for (var i = 0; i < numMoves; i++)
        {
            ref var move = ref moves[i];
            _state.Position.CalcFlip(ref move);
            _state.Update(ref move);
            values[i] = 1.0 - _valueFunc!.PredictWinRate<double>(_state.FeatureVector);
            _state.Undo(ref move);
        }

        var multiPV = new MultiPV();
        for (var i = 0; i < numMoves; i++)
        {
            multiPV.Add(new MultiPVItem([moves[i].Coord])
            {
                Depth = 0,
                EvalScore = (double)values[i],
                EvalScoreType = EvalScoreType.WinRate
            });
        }

        return multiPV;
    }
}