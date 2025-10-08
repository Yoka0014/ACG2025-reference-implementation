namespace ACG2025_reference_implementation.Learn;

using System;
using System.IO;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;
using System.Collections.Generic;
using System.Collections.Concurrent;

using ACG2025_reference_implementation.Reversi;
using ACG2025_reference_implementation.Evaluation;
using ACG2025_reference_implementation.Search.MCTS;

/// <summary>
/// Configuration record for the SelfPlayTrainer class.
/// Contains parameters for controlling the self-play training process.
/// </summary>
internal record class SelfPlayTrainerConfig
{
    /// <summary>
    /// Number of parallel actors for self-play data generation. Defaults to the number of processor cores.
    /// </summary>
    public int NumActors { get; init; } = Environment.ProcessorCount;
    
    /// <summary>
    /// Number of moves at the beginning of each game where moves are sampled from the visit count distribution.
    /// </summary>
    public int NumSamplingMoves { get; init; } = 30;
    
    /// <summary>
    /// Number of MCTS simulations to perform for each move decision.
    /// </summary>
    public int NumSimulations { get; init; } = 800;
    
    /// <summary>
    /// Alpha parameter for the Dirichlet noise added to the root node during MCTS search.
    /// </summary>
    public double RootDirichletAlpha { get; init; } = 0.3;
    
    /// <summary>
    /// Fraction of exploration noise to add to the root node during MCTS search.
    /// </summary>
    public double RootExplorationFraction { get; init; } = 0.25;
    
    /// <summary>
    /// Number of games to generate in each batch for training data.
    /// </summary>
    public int NumGamesInBatch { get; init; } = 500_000;
    
    /// <summary>
    /// Number of training epochs for each iteration.
    /// </summary>
    public int NumEpoch { get; init; } = 200;
    
    /// <summary>
    /// Whether to start the first iteration with random play data instead of self-play data.
    /// </summary>
    public bool StartWithRandomTrainData { get; init; } = true;
    
    /// <summary>
    /// Learning rate for the supervised training phase.
    /// </summary>
    public double LearningRate { get; init; } = (double)0.001;
    
    /// <summary>
    /// Base filename for saving the trained value function weights.
    /// </summary>
    public string WeightsFileName { get; init; } = "value_func_weights_sp";
}

/// <summary>
/// Self-play trainer that generates training data through self-play games using MCTS and trains a value function.
/// The trainer alternates between generating training data via self-play and updating the value function weights.
/// </summary>
/// <typeparam name="WeightType">The numeric type used for value function weights (must be a floating-point type).</typeparam>
internal class SelfPlayTrainer<WeightType> where WeightType : unmanaged, IFloatingPointIeee754<WeightType>
{
    /// <summary>
    /// Interval in milliseconds for showing progress logs during self-play data generation.
    /// </summary>
    const int ShowLogIntervalMs = 1000;

    /// <summary>
    /// Configuration for the self-play trainer.
    /// </summary>
    readonly SelfPlayTrainerConfig _config;
    
    /// <summary>
    /// Thread-safe queue containing the training dataset items generated from self-play games.
    /// </summary>
    readonly ConcurrentQueue<GameDatasetItem> _trainDataSet = new();
    
    /// <summary>
    /// Random number generator used for move sampling and other probabilistic decisions.
    /// </summary>
    readonly Random _rand;
    
    /// <summary>
    /// Array of random number generators, one for each actor to ensure thread safety.
    /// </summary>
    readonly Random[] _rands;
    
    /// <summary>
    /// Stream writer for logging training progress and statistics.
    /// </summary>
    readonly StreamWriter _logger;

    /// <summary>
    /// Initializes a new instance of the SelfPlayTrainer with the specified configuration.
    /// Uses the shared Random instance and null stream for logging.
    /// </summary>
    /// <param name="config">The configuration for the self-play trainer.</param>
    public SelfPlayTrainer(SelfPlayTrainerConfig config) : this(config, Random.Shared) { }

    /// <summary>
    /// Initializes a new instance of the SelfPlayTrainer with the specified configuration and random generator.
    /// Uses null stream for logging.
    /// </summary>
    /// <param name="config">The configuration for the self-play trainer.</param>
    /// <param name="rand">The random number generator to use.</param>
    public SelfPlayTrainer(SelfPlayTrainerConfig config, Random rand) : this(config, rand, Stream.Null) { }

    /// <summary>
    /// Initializes a new instance of the SelfPlayTrainer with the specified configuration, random generator, and log stream.
    /// </summary>
    /// <param name="config">The configuration for the self-play trainer.</param>
    /// <param name="rand">The random number generator to use.</param>
    /// <param name="logStream">The stream to write training logs to.</param>
    public SelfPlayTrainer(SelfPlayTrainerConfig config, Random rand, Stream logStream)
    {
        _config = config;
        _rand = rand;
        _rands = [.. Enumerable.Range(0, config.NumActors).Select(_ => new Random(rand.Next()))];
        _logger = new StreamWriter(logStream) { AutoFlush = false };
    }

    /// <summary>
    /// Finalizer that ensures the logger is properly disposed.
    /// </summary>
    ~SelfPlayTrainer() => _logger.Dispose();

    /// <summary>
    /// Trains the value function through iterative self-play and supervised learning.
    /// Each iteration generates new training data through self-play games and then trains the value function on this data.
    /// </summary>
    /// <param name="valueFunc">The value function to train.</param>
    /// <param name="numIterations">The number of training iterations to perform.</param>
    public void Train(ValueFunctionForTrain<WeightType> valueFunc, int numIterations)
    {
        for (var i = 0; i < numIterations; i++)
        {
            if (i == 0 && _config.StartWithRandomTrainData)
                GenerateTrainDataSetWithRandomPlay();
            else
                GenerateTrainDataSetWithSelfPlay(valueFunc);

            new SupervisedTrainer<WeightType>(valueFunc,
                new SupervisedTrainerConfig
                {
                    LearningRate = _config.LearningRate,
                    NumEpoch = _config.NumEpoch,
                    WeightsFileName = $"{_config.WeightsFileName}_{i}"
                }).Train([.. _trainDataSet], []);
        }
    }

    /// <summary>
    /// Generates training dataset by playing random games from the initial position.
    /// Used for the first iteration when StartWithRandomTrainData is true.
    /// </summary>
    void GenerateTrainDataSetWithRandomPlay()
    {
        _trainDataSet.Clear();
        for (var i = 0; i < _config.NumGamesInBatch; i++)
            _trainDataSet.Enqueue(GenerateRandomGame(new Position()));
    }

    /// <summary>
    /// Generates training dataset by playing self-play games using MCTS with the current value function.
    /// Uses multiple parallel actors to speed up data generation.
    /// </summary>
    /// <param name="valueFunc">The current value function to use for MCTS evaluations.</param>
    void GenerateTrainDataSetWithSelfPlay(ValueFunctionForTrain<WeightType> valueFunc)
    {
        _trainDataSet.Clear();

        var numActors = _config.NumActors;
        var quantValueFunc = ValueFunction.CreateFromTrainedValueFunction(valueFunc);

        var searchers = Enumerable.Range(0, numActors).Select(_ => new PUCTSearcher(quantValueFunc)
        {
            RootDirchletAlpha = _config.RootDirichletAlpha,
            RootExplorationFraction = _config.RootExplorationFraction
        }).ToArray();

        var numGamesPerActor = _config.NumGamesInBatch / numActors;

        _logger.WriteLine("Start self-play.");
        _logger.WriteLine($"The number of actors: {_config.NumActors}");
        _logger.WriteLine($"the number of MCTS simulations: {_config.NumSimulations}");
        _logger.Flush();

        var logTask = Task.Run(() =>
        {
            while (true)
            {
                var count = _trainDataSet.Count;
                _logger.WriteLine($"[{count}/{_config.NumGamesInBatch}]");
                _logger.Flush();

                if (count == _config.NumGamesInBatch)
                    break;

                Thread.Sleep(ShowLogIntervalMs);
            }
        }).ConfigureAwait(false);

        Parallel.For(0, numActors, actorID => genData(actorID, numGamesPerActor));

        var numGamesLeft = _config.NumGamesInBatch % numActors;
        genData(0, numGamesLeft);

        void genData(int actorID, int numGames)
        {
            var searcher = searchers[actorID];
            var rand = _rands[actorID];
            for (var i = 0; i < numGames; i++)
            {
                var data = GenerateTrainDataWithMCTS(new Position(), searcher, rand);
                _trainDataSet.Enqueue(data);
            }
        }
    }

    /// <summary>
    /// Generates a single training game by playing random moves from the given root position.
    /// </summary>
    /// <param name="rootPos">The starting position for the game.</param>
    /// <returns>A GameDatasetItem containing the game data for training.</returns>
    GameDatasetItem GenerateRandomGame(Position rootPos)
    {
        var rand = _rands[0];
        var pos = new Position(rootPos.GetBitboard(), rootPos.SideToMove);
        var moveHistory = new List<Move>();
        Span<Move> moves = stackalloc Move[Constants.MaxLegalMoves];

        var passCount = 0;
        while (passCount < 2)
        {
            var numMoves = pos.GetNextMoves(ref moves);

            if (numMoves == 0)
            {
                pos.Pass();
                moveHistory.Add(Move.Pass);
                passCount++;
                continue;
            }

            passCount = 0;
            ref var move = ref moves[rand.Next(numMoves)];
            pos.CalcFlip(ref move);
            pos.Update(ref move);
            moveHistory.Add(move);
        }

        moveHistory.RemoveRange(moveHistory.Count - 2, 2);  // removes last two passes.

        return new GameDatasetItem(rootPos, moveHistory, (sbyte)pos.GetScore(DiscColor.Black));
    }

    /// <summary>
    /// Generates a single training game using MCTS for move selection from the given root position.
    /// Records both the moves and the evaluation scores from MCTS for training data.
    /// </summary>
    /// <param name="rootPos">The starting position for the game.</param>
    /// <param name="searcher">The MCTS searcher to use for move selection.</param>
    /// <param name="rand">Random number generator for probabilistic move sampling.</param>
    /// <returns>A GameDatasetItem containing the game data and evaluation scores for training.</returns>
    GameDatasetItem GenerateTrainDataWithMCTS(Position rootPos, PUCTSearcher searcher, Random rand)
    {
        var pos = rootPos;
        var moveHistory = new List<Move>();
        var evalScores = new List<Half>();
        Span<Move> moves = stackalloc Move[Constants.MaxLegalMoves];
        var numSampleMoves = _config.NumSamplingMoves;

        searcher.SetRootPosition(ref pos);

        var passCount = 0;
        var moveCount = 0;
        while (passCount < 2)
        {
            var numMoves = pos.GetNextMoves(ref moves);

            if (numMoves == 0)
            {
                pos.Pass();
                moveHistory.Add(Move.Pass);
                evalScores.Add((Half)1 - evalScores[^1]);
                passCount++;

                if (!searcher.TransitionRootStateToChildState(BoardCoordinate.Pass))
                    searcher.SetRootPosition(ref pos);
                continue;
            }

            searcher.Search(_config.NumSimulations);

            Move? selectedMove;

            if (moveCount < numSampleMoves)
                selectedMove = searcher.SelectMoveWithVisitCountDist(_rand);
            else
                selectedMove = searcher.SelectBestMove();

            if (selectedMove is null)
                throw new InvalidOperationException("MCTS searcher failed to select a move. This may be caused by a low simulation count.");

            var move = selectedMove.Value;
            passCount = 0;
            pos.CalcFlip(ref move);
            pos.Update(ref move);
            moveHistory.Add(move);
            evalScores.Add((Half)searcher.RootValue);
            moveCount++;

            searcher.TransitionRootStateToChildState(move.Coord);
        }

        moveHistory.RemoveRange(moveHistory.Count - 2, 2);  // removes last two passes.
        evalScores.RemoveRange(evalScores.Count - 2, 2);

        return new GameDatasetItem(rootPos, moveHistory, evalScores, (sbyte)pos.GetScore(DiscColor.Black));
    }
}