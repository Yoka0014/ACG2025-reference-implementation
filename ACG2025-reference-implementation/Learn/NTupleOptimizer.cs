namespace ACG2025_reference_implementation.Learn;

using System;
using System.IO;
using System.Numerics;
using System.Linq;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

using ACG2025_reference_implementation.Reversi;
using ACG2025_reference_implementation.NTupleSystem;
using ACG2025_reference_implementation.Evaluation;
using ACG2025_reference_implementation.Search.MCTS;
using ACG2025_reference_implementation.Utils;

/// <summary>
/// Configuration parameters for the n-tuple optimizer.
/// Contains settings that control the optimization process including data generation,
/// self-play parameters, and training intervals.
/// </summary>
internal record class NTupleOptimizerConfig
{
    /// <summary>
    /// Gets or sets the number of threads to use for parallel processing during optimization.
    /// Defaults to the number of processor cores available on the system.
    /// </summary>
    public int NumThreads { get; init; } = Environment.ProcessorCount;

    /// <summary>
    /// Gets or sets the number of training data samples to generate.
    /// </summary>
    public int NumTrainData { get; init; } = 10000;
    
    /// <summary>
    /// Gets or sets the number of test data samples to generate.
    /// </summary>
    public int NumTestData { get; init; } = 10000;
    
    /// <summary>
    /// Gets or sets the variation factor for training data generation.
    /// Controls the tolerance for selecting near-optimal moves during self-play.
    /// </summary>
    public double TrainDataVariationFactor{ get; init; } = 0.05;
    
    /// <summary>
    /// Gets or sets the number of MCTS simulations per move during self-play.
    /// </summary>
    public uint NumSimulations { get; init; } = 3200;
    
    /// <summary>
    /// Gets or sets the interval (in generations) for updating training data.
    /// </summary>
    public int TrainDataUpdateInterval { get; init; } = 100;
}

/// <summary>
/// Implements an iterative n-tuple optimization system that combines genetic algorithms, 
/// reinforcement learning, and Monte Carlo Tree Search to evolve optimal n-tuple configurations 
/// for Reversi position evaluation.
/// 
/// The optimization process follows these key phases:
/// 1. Generate initial training data from random games
/// 2. Evolve n-tuple configurations using BRKGA (Biased Random-Key Genetic Algorithm)
/// 3. Train value functions using temporal difference learning
/// 4. Generate new training data using MCTS with trained agents
/// 5. Repeat steps 2-4 iteratively to improve n-tuple quality
/// </summary>
/// <typeparam name="WeightType">The floating-point type used for value function weights (Half, float, or double)</typeparam>
internal class NTupleOptimizer<WeightType> where WeightType : unmanaged, IFloatingPointIeee754<WeightType>
{
    /// <summary>Configuration parameters for the optimizer</summary>
    readonly NTupleOptimizerConfig _config;
    
    /// <summary>Configuration parameters for the genetic algorithm</summary>
    readonly BRKGAConfig _gaConfig;
    
    /// <summary>Configuration parameters for temporal difference learning</summary>
    readonly TDTrainerConfig _tdConfig;
    
    /// <summary>Size of each n-tuple (number of board coordinates)</summary>
    readonly int _nTupleSize;
    
    /// <summary>Number of n-tuples per individual</summary>
    readonly int _numNTuples;
    
    /// <summary>Main random number generator</summary>
    readonly Random _rand;

    /// <summary>Array of random number generators for parallel processing</summary>
    Random[] _rands = [];
    
    /// <summary>Parallel processing configuration</summary>
    ParallelOptions _parallelOptions = new();

    Individual[]? _currentPool;

    /// <summary>
    /// Initializes a new instance of NTupleOptimizer with the specified configuration parameters.
    /// Uses a shared random number generator.
    /// </summary>
    /// <param name="config">Configuration parameters for the optimizer</param>
    /// <param name="gaConfig">Configuration parameters for the genetic algorithm</param>
    /// <param name="tdConfig">Configuration parameters for temporal difference learning</param>
    /// <param name="nTupleSize">Size of each n-tuple (number of board coordinates)</param>
    /// <param name="numNTuples">Number of n-tuples per individual</param>
    public NTupleOptimizer(NTupleOptimizerConfig config, BRKGAConfig gaConfig, TDTrainerConfig tdConfig, int nTupleSize, int numNTuples)
    : this(config, gaConfig, tdConfig, nTupleSize, numNTuples, Random.Shared) { }

    /// <summary>
    /// Initializes a new instance of NTupleOptimizer with the specified configuration parameters and random generator.
    /// </summary>
    /// <param name="config">Configuration parameters for the optimizer</param>
    /// <param name="gaConfig">Configuration parameters for the genetic algorithm</param>
    /// <param name="tdConfig">Configuration parameters for temporal difference learning</param>
    /// <param name="nTupleSize">Size of each n-tuple (number of board coordinates)</param>
    /// <param name="numNTuples">Number of n-tuples per individual</param>
    /// <param name="rand">Random number generator for genetic operations and game generation</param>
    public NTupleOptimizer(NTupleOptimizerConfig config, BRKGAConfig gaConfig, TDTrainerConfig tdConfig, int nTupleSize, int numNTuples, Random rand)
    {
        _config = config;
        _gaConfig = gaConfig;
        _tdConfig = tdConfig;
        _nTupleSize = nTupleSize;
        _numNTuples = numNTuples;
        _rand = rand;
        NumThreads = Environment.ProcessorCount;
    }

    /// <summary>
    /// Gets or sets the maximum number of threads used for parallel processing.
    /// When set, creates dedicated random number generators for each thread.
    /// </summary>
    public int NumThreads
    {
        get => _parallelOptions.MaxDegreeOfParallelism;

        set
        {
            _parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = value };
            _rands = [.. Enumerable.Range(0, value).Select(_ => new Random(_rand.Next()))];
        }
    }

    /// <summary>
    /// Starts the complete n-tuple optimization process from scratch.
    /// Generates initial training data from random games, then iteratively improves 
    /// n-tuple configurations through genetic evolution and reinforcement learning.
    /// </summary>
    /// <param name="numGenerations">Total number of genetic algorithm generations to run</param>
    public void Train(int numGenerations)
    {
        var gaConfig = _gaConfig with
        {
            PoolFileName = $"{_gaConfig.PoolFileName}_0_",
            FitnessHistoryFileName = $"{_gaConfig.FitnessHistoryFileName}_0"
        };

        var ga = new BRKGA<WeightType>(gaConfig, _nTupleSize, _numNTuples);

        Console.WriteLine("Generate random play games.");

        var trainData = GenerateTrainDataFromRandomGame(_config.NumTrainData);
        var testData = GenerateTrainDataFromRandomGame(_config.NumTestData);

        Console.WriteLine("Start n-tuple optimization with random play games.");

        var numGens = Math.Min(_config.TrainDataUpdateInterval, numGenerations);
        ga.Train(trainData, testData, numGens);
        StartTrainLoop(numGenerations - numGens, ga.GetCurrentPool());
    }

    /// <summary>
    /// Continues the optimization process from a previously saved genetic algorithm population.
    /// </summary>
    /// <param name="poolPath">Path to the file containing the saved population pool</param>
    /// <param name="numGenerations">Number of additional generations to run</param>
    public void Train(string poolPath, int numGenerations)
    {
        var pool = Individual.LoadPoolFromFile(poolPath);
        StartTrainLoop(numGenerations, pool);
    }

    /// <summary>
    /// Executes the main training loop that alternates between reinforcement learning, 
    /// MCTS-based data generation, and genetic evolution of n-tuple configurations.
    /// </summary>
    /// <param name="numGenerations">Number of generations remaining to process</param>
    /// <param name="initialPool">Initial population of genetic algorithm individuals</param>
    void StartTrainLoop(int numGenerations, Individual[] initialPool)
    {
        var pool = initialPool;
        var numElites = (int)(_gaConfig.EliteRate * _gaConfig.PopulationSize);
        var genLeft = numGenerations;

        var id = 1;
        while (genLeft > 0)
        {
            Console.WriteLine($"\nGenerations left: {genLeft}");

            var gaConfig = _gaConfig with
            {
                PoolFileName = $"{_gaConfig.PoolFileName}_{id}_",
                FitnessHistoryFileName = $"{_gaConfig.FitnessHistoryFileName}_{id}_"
            };
            var ga = new BRKGA<WeightType>(gaConfig, _nTupleSize, _numNTuples);

            var nTupleManagers = BRKGA<WeightType>.DecodePool(pool, _nTupleSize, _numNTuples)[..numElites];

            Console.WriteLine("Start RL.");
            var valueFuncs = TrainAgents(nTupleManagers);

            Console.WriteLine("Generate train data with MCTS.");
            var quantValueFuncs = valueFuncs.Select(vf => ValueFunction.CreateFromTrainedValueFunction(vf)).ToArray();
            var trainData = GenerateTrainDataWithMCTS(_config.NumTrainData, quantValueFuncs);

            Console.WriteLine("Generate test data with MCTS.");
            var testData = GenerateTrainDataWithMCTS(_config.NumTestData, quantValueFuncs);

            Console.WriteLine("Start n-tuple optimization.");
            var numGens = Math.Min(_config.TrainDataUpdateInterval, genLeft);
            ga.Train(pool, trainData, testData, numGens);
            pool = ga.GetCurrentPool();
            genLeft -= numGens;
            id++;
        }
    }

    /// <summary>
    /// Trains value functions using temporal difference learning.
    /// Each n-tuple manager is converted to a trainable value function and trained 
    /// using self-play with the TD learning algorithm.
    /// </summary>
    /// <param name="nTupleManagers">Array of n-tuple managers to convert and train</param>
    /// <returns>Array of trained value functions ready for MCTS-based game generation</returns>
    ValueFunctionForTrain<WeightType>[] TrainAgents(NTupleManager[] nTupleManagers)
    {
        var valueFuncs = nTupleManagers.Select(nt => new ValueFunctionForTrain<WeightType>(nt)).ToArray();

        for (var i = 0; i < valueFuncs.Length; i += NumThreads)
        {
            var numThreads = Math.Min(NumThreads, valueFuncs.Length - i);
            Parallel.For(0, numThreads, _parallelOptions, threadID =>
            {
                var trainer = new TDTrainer<WeightType>($"AG-{i + threadID}", valueFuncs[i + threadID], _tdConfig, Stream.Null, _rands[threadID]);
                trainer.Train();
            });  
        }

        return valueFuncs;
    }

    /// <summary>
    /// Generates training data by playing games between trained agents using MCTS.
    /// Agents are randomly paired to play against each other, creating diverse 
    /// training positions with corresponding game outcomes.
    /// </summary>
    /// <param name="numData">Number of games to generate</param>
    /// <param name="valueFuncs">Array of trained value functions to use as players</param>
    /// <returns>Dataset containing game positions and outcomes for supervised learning</returns>
    GameDataset GenerateTrainDataWithMCTS(int numData, ValueFunction[] valueFuncs)
    {
        var trainData = new GameDatasetItem[numData];
        var numThreads = NumThreads;
        var numGamesPerThread = numData / numThreads;
        var count = 0;
        Parallel.For(0, numThreads, _parallelOptions,
            threadID => Worker(threadID, trainData.AsSpan(numGamesPerThread * threadID, numGamesPerThread)));

        Worker(0, trainData.AsSpan(numGamesPerThread * numThreads, numData % numThreads));

        return trainData;

        void Worker(int threadID, Span<GameDatasetItem> data)
        {
            var rand = _rands[threadID];
            var ag0 = rand.Next(valueFuncs.Length);
            int ag1;
            do
                ag1 = rand.Next(valueFuncs.Length);
            while (ag0 == ag1);

            for (var i = 0; i < data.Length; i++)
            {
                data[i] = GenerateGameWithMCTS(threadID, new Position(), valueFuncs[ag0], valueFuncs[ag1]);
                Interlocked.Increment(ref count);
                Console.WriteLine($"[{count} / {numData}]");
            }
        }
    }

    /// <summary>
    /// Generates training data from completely random games.
    /// Used for initial training when no strong players are available yet.
    /// </summary>
    /// <param name="numData">Number of random games to generate</param>
    /// <returns>Dataset containing random game positions and outcomes</returns>
    GameDataset GenerateTrainDataFromRandomGame(int numData)
        => [.. Enumerable.Range(0, numData).Select(_ => GenerateRandomGame(new Position()))];

    /// <summary>
    /// Generates a single game using Monte Carlo Tree Search between two value functions.
    /// Uses PUCT algorithm for move selection with some variation to ensure diverse training data.
    /// The game continues until both players pass consecutively.
    /// </summary>
    /// <param name="threadID">Thread identifier for accessing thread-local random generator</param>
    /// <param name="rootPos">Starting position for the game</param>
    /// <param name="valueFuncForBlack">Value function for the black player</param>
    /// <param name="valueFuncForWhite">Value function for the white player</param>
    /// <returns>Complete game record including moves and final outcome</returns>
    GameDatasetItem GenerateGameWithMCTS(int threadID, Position rootPos, ValueFunction valueFuncForBlack, ValueFunction valueFuncForWhite)
    {
        var rand = _rands[threadID];
        var pos = rootPos;
        var moveHistory = new List<Move>();
        Span<Move> moves = stackalloc Move[Constants.MaxLegalMoves];
        var moveEvals = new MoveEval[Constants.MaxLegalMoves];

        var searcher = new PUCTSearcher(valueFuncForBlack);
        var oppSearcher = new PUCTSearcher(valueFuncForWhite);
        searcher.SetRootPosition(ref pos);
        oppSearcher.SetRootPosition(ref pos);
        var numMoves = pos.GetNextMoves(ref moves);
        Span<double> moveSelectionProb = stackalloc double[Constants.MaxLegalMoves];

        var passCount = 0;
        while (passCount < 2)
        {
            if (numMoves == 0)
            {
                pos.Pass();
                numMoves = pos.GetNextMoves(ref moves);
                moveHistory.Add(Move.Pass);
                passCount++;

                UpdateSearcherState(searcher, ref pos, BoardCoordinate.Pass);
                UpdateSearcherState(oppSearcher, ref pos, BoardCoordinate.Pass);
                (searcher, oppSearcher) = (oppSearcher, searcher);
                continue;
            }

            passCount = 0;
            Move move;
            if (numMoves == 1)
            {
                move = moves[0];
            }
            else
            {
                searcher.SearchOnSingleThread(_config.NumSimulations);
                var searchRes = searcher.GetSearchResult();

                searchRes!.ChildEvals.CopyTo(moveEvals);
                var bestValue = moveEvals[0].ExpectedReward;
                var numCandidates = 0;
                var simulationCount = 0L;
                for (var i = 0; i < numMoves; i++)
                {
                    if (bestValue - moveEvals[i].ExpectedReward <= _config.TrainDataVariationFactor)
                    {
                        (moveEvals[numCandidates], moveEvals[i]) = (moveEvals[i], moveEvals[numCandidates]);
                        simulationCount += moveEvals[numCandidates].SimulationCount;
                        numCandidates++;
                    }
                }

                rand.Shuffle(moveEvals.AsSpan(0, numCandidates));

                var prob = moveSelectionProb[..numCandidates];
                for (var i = 0; i < prob.Length; i++)
                    prob[i] = moveEvals[i].Effort;

                var idx = rand.Sample(prob);
                move = new Move(moveEvals[idx].Move);
            }

            pos.CalcFlip(ref move);
            pos.Update(ref move);
            numMoves = pos.GetNextMoves(ref moves);

            UpdateSearcherState(searcher, ref pos, move.Coord);
            UpdateSearcherState(oppSearcher, ref pos, move.Coord);
            (searcher, oppSearcher) = (oppSearcher, searcher);
            moveHistory.Add(move);
        }

        moveHistory.RemoveRange(moveHistory.Count - 2, 2);  // removes last two passes.

        return new GameDatasetItem(rootPos, moveHistory, (sbyte)pos.GetScore(DiscColor.Black));

        /// <summary>
        /// Updates the MCTS searcher state to reflect a move that was played.
        /// If the searcher can transition to a child node representing the move, it does so;
        /// otherwise, it resets the searcher to the new position.
        /// </summary>
        /// <param name="searcher">The PUCT searcher to update</param>
        /// <param name="pos">The current game position</param>
        /// <param name="move">The move that was played</param>
        static void UpdateSearcherState(PUCTSearcher searcher, ref Position pos, BoardCoordinate move)
        {
            if (!searcher.TransitionRootStateToChildState(move))
                searcher.SetRootPosition(ref pos);
        }
    }

    /// <summary>
    /// Generates a single game with completely random move selection.
    /// Used for creating initial training data when no trained players are available.
    /// </summary>
    /// <param name="rootPos">Starting position for the game</param>
    /// <returns>Complete game record with random moves and final outcome</returns>
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
}