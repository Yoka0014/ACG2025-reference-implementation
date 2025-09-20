namespace ACG2025_reference_implementation.Learn;

using System;
using System.Numerics;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Collections;
using System.Diagnostics;
using System.Threading.Tasks;
using System.Runtime.CompilerServices;

using ACG2025_reference_implementation.Reversi;
using ACG2025_reference_implementation.NTupleSystem;
using ACG2025_reference_implementation.Evaluation;
using ACG2025_reference_implementation.Search;
using ACG2025_reference_implementation.Utils;

/// <summary>
/// Configuration parameters for temporal difference learning.
/// Contains hyperparameters and settings for the TD learning algorithm.
/// </summary>
internal record class TDTrainerConfig
{
    /// <summary>
    /// Gets or sets the number of training episodes to run.
    /// </summary>
    public int NumEpisodes { get; init; } = 250_000;
    
    /// <summary>
    /// Gets or sets the number of initial random moves in each episode.
    /// </summary>
    public int NumInitialRandomMoves { get; init; } = 1;
    
    /// <summary>
    /// Gets or sets the learning rate for weight updates.
    /// </summary>
    public double LearningRate { get; init; } = 0.2;
    
    /// <summary>
    /// Gets or sets the discount factor for future rewards.
    /// </summary>
    public double DiscountRate { get; init; } = 1.0;
    
    /// <summary>
    /// Gets or sets the initial exploration rate for epsilon-greedy policy.
    /// </summary>
    public double InitialExplorationRate { get; init; } = 0.2;
    
    /// <summary>
    /// Gets or sets the final exploration rate for epsilon-greedy policy.
    /// </summary>
    public double FinalExplorationRate { get; init; } = 0.1;
    
    /// <summary>
    /// Gets or sets the eligibility trace decay factor (lambda).
    /// </summary>
    public double EligibilityTraceFactor { get; init; } = 0.5;
    
    /// <summary>
    /// Gets or sets the horizon cut factor for limiting eligibility trace length.
    /// </summary>
    public double HorizonCutFactor { get; init; } = 0.1;
    
    /// <summary>
    /// Gets or sets the TCL (Temporal Coherence Learning) factor (beta).
    /// </summary>
    public double TCLFactor { get; init; } = 2.7;

    /// <summary>
    /// Gets or sets the base filename for saving trained weights.
    /// </summary>
    public string WeightsFileName { get; init; } = "value_func_weights_td";
    
    /// <summary>
    /// Gets or sets the interval (in episodes) for saving weights to disk.
    /// </summary>
    public int SaveWeightsInterval { get; init; } = 10000;
    
    /// <summary>
    /// Gets or sets whether to save only the latest weights (overwriting previous saves).
    /// </summary>
    public bool SaveOnlyLatestWeights { get; init; } = true;
}

/// <summary>
/// Implements temporal difference learning for training value functions in game playing.
/// Uses eligibility traces and TCL for efficient learning.
/// </summary>
/// <typeparam name="WeightType">The floating-point type used for weights (Half, float, or double)</typeparam>
internal class TDTrainer<WeightType> where WeightType : unmanaged, IFloatingPointIeee754<WeightType>
{
    /// <summary>
    /// Small epsilon value used in TCL calculations.
    /// </summary>
    const double TCLEpsilon = 1.0e-4;

    /// <summary>
    /// Gets the label identifier for this trainer instance.
    /// </summary>
    public string Label { get; }
    
    readonly TDTrainerConfig _config;
    readonly double _explorationRateDiff;
    readonly string _weightsFilePath;
    readonly StreamWriter _logger;

    readonly ValueFunctionForTrain<WeightType> _valueFunc;
    readonly PastStatesBuffer _pastStatesBuffer;
    readonly WeightType[] _weightDiffSum;
    readonly WeightType[] _weightAbsDiffSum;
    readonly WeightType[] _biasDiffSum;
    readonly WeightType[] _biasAbsDiffSum;
    readonly WeightType _tclFactor;

    readonly Random _rand;

    /// <summary>
    /// Initializes a new instance of TDTrainer with console output logging.
    /// </summary>
    /// <param name="valueFunc">The value function to train</param>
    /// <param name="config">Training configuration parameters</param>
    /// <param name="randSeed">Random seed for reproducibility (-1 for random seed)</param>
    public TDTrainer(ValueFunctionForTrain<WeightType> valueFunc, TDTrainerConfig config, int randSeed = -1)
    : this(valueFunc, config, Console.OpenStandardOutput(), randSeed) { }

    /// <summary>
    /// Initializes a new instance of TDTrainer with custom output stream.
    /// </summary>
    /// <param name="valueFunc">The value function to train</param>
    /// <param name="config">Training configuration parameters</param>
    /// <param name="logStream">Output stream for logging</param>
    /// <param name="randSeed">Random seed for reproducibility (-1 for random seed)</param>
    public TDTrainer(ValueFunctionForTrain<WeightType> valueFunc, TDTrainerConfig config, Stream logStream, int randSeed = -1)
    : this(string.Empty, valueFunc, config, logStream, randSeed) { }

    /// <summary>
    /// Initializes a new instance of TDTrainer with label and console output.
    /// </summary>
    /// <param name="label">Label identifier for this trainer</param>
    /// <param name="valueFunc">The value function to train</param>
    /// <param name="config">Training configuration parameters</param>
    public TDTrainer(string label, ValueFunctionForTrain<WeightType> valueFunc, TDTrainerConfig config)
    : this(label, valueFunc, config, Console.OpenStandardOutput()) { }

    /// <summary>
    /// Initializes a new instance of TDTrainer with full customization.
    /// </summary>
    /// <param name="label">Label identifier for this trainer</param>
    /// <param name="valueFunc">The value function to train</param>
    /// <param name="config">Training configuration parameters</param>
    /// <param name="logStream">Output stream for logging</param>
    /// <param name="randSeed">Random seed for reproducibility (-1 for random seed)</param>
    public TDTrainer(string label, ValueFunctionForTrain<WeightType> valueFunc, TDTrainerConfig config, Stream logStream, int randSeed = -1)
    {
        Label = label;
        _config = config;
        _explorationRateDiff = (config.InitialExplorationRate - config.FinalExplorationRate) / config.NumEpisodes;
        _weightsFilePath = $"{config.WeightsFileName}{"{0}"}.bin";

        _valueFunc = valueFunc;
        var numWeights = valueFunc.Weights.Length / 2;
        _weightDiffSum = new WeightType[numWeights];
        _weightAbsDiffSum = new WeightType[numWeights];
        _biasDiffSum = new WeightType[valueFunc.NumPhases];
        _biasAbsDiffSum = new WeightType[valueFunc.NumPhases];
        _tclFactor = WeightType.CreateChecked(_config.TCLFactor);
        var capacity = int.CreateChecked(Math.Log(config.HorizonCutFactor, config.EligibilityTraceFactor)) + 1;
        _pastStatesBuffer = new PastStatesBuffer(capacity, valueFunc.NTupleManager);

        _rand = (randSeed >= 0) ? new Random(randSeed) : new Random(Random.Shared.Next());

        _logger = new StreamWriter(logStream) { AutoFlush = false };
    }

    /// <summary>
    /// Trains multiple agents in parallel using the specified configuration.
    /// Uses all available processor cores for parallel training.
    /// </summary>
    /// <param name="config">Training configuration parameters</param>
    /// <param name="numAgents">Number of agents to train in parallel</param>
    /// <param name="nTupleSize">Size of each n-tuple</param>
    /// <param name="numNTuples">Number of n-tuples per agent</param>
    /// <param name="numMovesPerPhase">Number of moves per game phase</param>
    public static void TrainMultipleAgents(TDTrainerConfig config, int numAgents, int nTupleSize, int numNTuples, int numMovesPerPhase)
        => TrainMultipleAgents(config, numAgents, nTupleSize, numNTuples, numMovesPerPhase, Environment.ProcessorCount);

    /// <summary>
    /// Trains multiple agents in parallel using the specified configuration and thread count.
    /// </summary>
    /// <param name="config">Training configuration parameters</param>
    /// <param name="numAgents">Number of agents to train in parallel</param>
    /// <param name="nTupleSize">Size of each n-tuple</param>
    /// <param name="numNTuples">Number of n-tuples per agent</param>
    /// <param name="numMovesPerPhase">Number of moves per game phase</param>
    /// <param name="numThreads">Maximum number of threads to use for parallel training</param>
    public static void TrainMultipleAgents(TDTrainerConfig config, int numAgents, int nTupleSize, int numNTuples, int numMovesPerPhase, int numThreads)
    {
        var options = new ParallelOptions { MaxDegreeOfParallelism = numThreads };
        Parallel.For(0, numAgents, options, agentID =>
        {
            var dir = $"AG-{agentID}";
            if (!Directory.Exists(dir))
                Directory.CreateDirectory(dir);

            var nTuples = (from _ in Enumerable.Range(0, numNTuples) select new NTuple(nTupleSize)).ToArray();
            var nTupleManager = new NTupleManager(nTuples);
            var valueFunc = new ValueFunctionForTrain<WeightType>(nTupleManager, numMovesPerPhase);
            new TDTrainer<WeightType>($"AG-{agentID}", valueFunc, config with { WeightsFileName = Path.Combine(dir, config.WeightsFileName) }).Train();
        });
    }

    /// <summary>
    /// Runs the complete training process for the specified number of episodes.
    /// Implements temporal difference learning with eligibility traces and exploration decay.
    /// </summary>
    public void Train()
    {
        var explorationRate = _config.InitialExplorationRate;
        var tclEpsilon = WeightType.CreateChecked(TCLEpsilon);
        Array.Fill(_weightDiffSum, tclEpsilon);
        Array.Fill(_weightAbsDiffSum, tclEpsilon);
        Array.Fill(_biasDiffSum, tclEpsilon);
        Array.Fill(_biasAbsDiffSum, tclEpsilon);

        PrintLabel();
        _logger.WriteLine("Start learning.\n");
        PrintParams(explorationRate);
        _logger.Flush();

        for (var episodeID = 0; episodeID < _config.NumEpisodes; episodeID++)
        {
            RunEpisode(explorationRate);
            explorationRate -= _explorationRateDiff;

            if ((episodeID + 1) % _config.SaveWeightsInterval == 0)
            {
                PrintLabel();

                var fromEpisodeID = episodeID - _config.SaveWeightsInterval + 1;
                _logger.WriteLine($"Episodes {fromEpisodeID} to {episodeID} have done.");

                var suffix = _config.SaveOnlyLatestWeights ? string.Empty : $"_{episodeID}";
                var path = string.Format(_weightsFilePath, suffix);
                _valueFunc.SaveToFile(path);

                _logger.WriteLine($"Weights were saved at \"{path}\"\n");
                PrintParams(explorationRate);
                _logger.WriteLine();
                _logger.Flush();
            }
        }

        _valueFunc.CopyWeightsBlackToWhite();
    }

    /// <summary>
    /// Prints the trainer label to the log output if available.
    /// </summary>
    void PrintLabel()
    {
        if (!string.IsNullOrEmpty(Label))
            _logger.WriteLine($"[{Label}]");
    }

    /// <summary>
    /// Prints current training parameters to the log output.
    /// </summary>
    /// <param name="explorationRate">Current exploration rate</param>
    void PrintParams(double explorationRate)
    {
        _logger.WriteLine($"Lambda: {_config.EligibilityTraceFactor}");
        _logger.WriteLine($"ExplorationRate: {explorationRate}");
        _logger.WriteLine($"MeanLearningRate: {CalcMeanLearningRate()}");
    }

    /// <summary>
    /// Calculates the mean learning rate across all weights and biases using TCL adaptive rates.
    /// </summary>
    /// <returns>The mean learning rate</returns>
    WeightType CalcMeanLearningRate()
    {
        var sum = WeightType.Zero;
        foreach ((var n, var a) in _weightDiffSum.Zip(_weightAbsDiffSum))
            sum += Decay(WeightType.Abs(n / a));

        foreach ((var n, var a) in _biasDiffSum.Zip(_biasAbsDiffSum))
            sum += Decay(WeightType.Abs(n / a));

        var lr = WeightType.CreateChecked(_config.LearningRate);
        return lr * sum / WeightType.CreateChecked(_weightDiffSum.Length + _biasDiffSum.Length);
    }

    /// <summary>
    /// Runs a single training episode using temporal difference learning.
    /// Combines random exploration with greedy action selection based on the current exploration rate.
    /// </summary>
    /// <param name="explorationRate">Current exploration rate for epsilon-greedy policy</param>
    void RunEpisode(double explorationRate)
    {
        var gamma = WeightType.CreateChecked(_config.DiscountRate);
        var state = new State(_valueFunc.NTupleManager);
        _pastStatesBuffer.Clear();
        _pastStatesBuffer.Add(state.FeatureVector);
        Span<Move> moves = stackalloc Move[Constants.MaxLegalMoves];
        int numMoves = state.Position.GetNextMoves(ref moves);

        var moveCount = 0;
        var afterPass = false;
        while (true)
        {
            WeightType v = _valueFunc.PredictWithBlackWeights(state.FeatureVector);
            if (numMoves == 0) // pass
            {
                if (afterPass)
                {
                    Adapt((WeightType.One - GetReward(state.Position.DiscDiff)) - v);
                    break;
                }

                afterPass = true;
                state.Pass();
                numMoves = state.Position.GetNextMoves(ref moves);
                _pastStatesBuffer.Add(state.FeatureVector);
                continue;
            }

            WeightType nextV;
            if (moveCount < _config.NumInitialRandomMoves || _rand.NextDouble() < explorationRate)   // random move
            {
                ref var move = ref moves[_rand.Next(numMoves)];
                state.Position.CalcFlip(ref move);
                state.Update(ref move);
                numMoves = state.Position.GetNextMoves(ref moves);
                nextV = WeightType.One - _valueFunc.PredictWithBlackWeights(state.FeatureVector);
            }
            else    // greedy
            {
                ref Move bestMove = ref moves[0];
                var minVLogit = WeightType.PositiveInfinity;
                for (var i = 0; i < numMoves; i++)
                {
                    ref Move move = ref moves[i];
                    state.Position.CalcFlip(ref move);
                    state.Update(ref move);

                    WeightType vLogit = _valueFunc.PredictLogitWithBlackWeights(state.FeatureVector);

                    if (vLogit < minVLogit)
                    {
                        minVLogit = vLogit;
                        bestMove = ref move;
                    }

                    state.Undo(ref move);
                }

                state.Update(ref bestMove);
                numMoves = state.Position.GetNextMoves(ref moves);
                nextV = WeightType.One - MathFunctions.StdSigmoid(minVLogit);
            }

            Adapt(gamma * nextV - v);
            _pastStatesBuffer.Add(state.FeatureVector);
            moveCount++;
        }
    }

    /// <summary>
    /// Applies temporal difference error updates to weights and biases using eligibility traces.
    /// Implements TCL for adaptive learning rates.
    /// </summary>
    /// <param name="tdError">The temporal difference error to propagate</param>
    unsafe void Adapt(WeightType tdError)
    {
        var gamma = WeightType.CreateChecked(_config.DiscountRate);
        var lambda = WeightType.CreateChecked(_config.EligibilityTraceFactor);
        var alpha = WeightType.CreateChecked(_config.LearningRate);
        var beta = WeightType.CreateChecked(_config.TCLFactor);
        var eligibilityFactor = WeightType.One;

        fixed (WeightType* weights = _valueFunc.Weights)
        fixed (WeightType* bias = _valueFunc.Bias)
        fixed (WeightType* weightDiffSum = _weightDiffSum)
        fixed (WeightType* weightAbsDiffSum = _weightAbsDiffSum)
        fixed (WeightType* biasDiffSum = _biasDiffSum)
        fixed (WeightType* biasAbsDiffSum = _biasAbsDiffSum)
        {
            foreach (var posFeatureVec in _pastStatesBuffer)
            {
                int phase;
                fixed (int* toPhase = _valueFunc.EmptyCellCountToPhase)
                    phase = toPhase[posFeatureVec.EmptyCellCount];

                var delta = eligibilityFactor * tdError;
                if (posFeatureVec.SideToMove == DiscColor.Black)
                    ApplyGradients<Black>(phase, posFeatureVec, weights, weightDiffSum, weightAbsDiffSum, alpha, beta, delta);
                else
                    ApplyGradients<White>(phase, posFeatureVec, weights, weightDiffSum, weightAbsDiffSum, alpha, beta, delta);

                var reg = WeightType.One / WeightType.CreateChecked(posFeatureVec.NumNTuples + 1);
                var lr = reg * alpha * Decay(WeightType.Abs(biasDiffSum[phase]) / biasAbsDiffSum[phase]);
                var db = lr * delta;
                bias[phase] += db;
                biasDiffSum[phase] += db;
                biasAbsDiffSum[phase] += WeightType.Abs(db);

                eligibilityFactor *= gamma * lambda;
                tdError = lambda - WeightType.One - tdError;  // inverse tdError: lambda * (1.0 - nextV) - (1.0 - v)
            }
        }
    }

    /// <summary>
    /// Applies gradient updates to n-tuple weights with mirror symmetry handling.
    /// Uses TCL for adaptive learning rates based on weight update history.
    /// </summary>
    /// <typeparam name="DiscColor">The disc color type (Black or White)</typeparam>
    /// <param name="phase">Game phase index</param>
    /// <param name="featureVec">Feature vector for the current position</param>
    /// <param name="weights">Pointer to the weights array</param>
    /// <param name="weightDeltaSum">Pointer to cumulative weight changes</param>
    /// <param name="weightDeltaAbsSum">Pointer to cumulative absolute weight changes</param>
    /// <param name="alpha">Base learning rate</param>
    /// <param name="beta">TCL factor for adaptive learning rate</param>
    /// <param name="delta">Weight update delta value</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    unsafe void ApplyGradients<DiscColor>(int phase, FeatureVector featureVec, WeightType* weights, WeightType* weightDeltaSum, WeightType* weightDeltaAbsSum, WeightType alpha, WeightType beta, WeightType delta) where DiscColor : IDiscColor
    {
        for (var i = 0; i < featureVec.Features.Length; i++)
        {
            var offset = _valueFunc.PhaseOffset[phase] + _valueFunc.NTupleOffset[i];
            var w = weights + offset;
            var dwSum = weightDeltaSum + offset;
            var dwAbsSum = weightDeltaAbsSum + offset;

            ref Feature feature = ref featureVec.Features[i];
            fixed (int* opp = featureVec.NTupleManager.GetRawOpponentFeatureTable(i))
            fixed (int* mirror = featureVec.NTupleManager.GetRawMirroredFeatureTable(i))
            {
                var reg = WeightType.One / WeightType.CreateChecked((featureVec.NumNTuples + 1) * feature.Length);
                for (var j = 0; j < feature.Length; j++)
                {
                    var f = (typeof(DiscColor) == typeof(Black)) ? feature[j] : opp[feature[j]];
                    var mf = mirror[f];

                    var lr = reg * alpha * Decay(WeightType.Abs(dwSum[f]) / dwAbsSum[f]);
                    var dw = lr * delta;
                    var absDW = WeightType.Abs(dw);

                    if (mf != f)
                    {
                        dw *= WeightType.CreateChecked(0.5);
                        absDW *= WeightType.CreateChecked(0.5);
                        w[mf] += dw;
                        dwSum[mf] += dw;
                        dwAbsSum[mf] += absDW;
                    }

                    w[f] += dw;
                    dwSum[f] += dw;
                    dwAbsSum[f] += absDW;
                }
            }
        }
    }

    /// <summary>
    /// Applies exponential decay function used in TCL.
    /// </summary>
    /// <param name="x">Input value</param>
    /// <returns>Exponentially decayed value</returns>
    WeightType Decay(WeightType x) => WeightType.Exp(_tclFactor * (x - WeightType.One));

    /// <summary>
    /// Converts the final disc difference to a reward value.
    /// </summary>
    /// <param name="discDiff">Final disc count difference (positive favors current player)</param>
    /// <returns>Reward value: 1.0 for win, 0.0 for loss, 0.5 for draw</returns>
    static WeightType GetReward(int discDiff)
    {
        if (discDiff == 0)
            return WeightType.CreateChecked(0.5);
        else
            return (discDiff > 0) ? WeightType.One : WeightType.Zero;
    }

    /// <summary>
    /// A circular buffer that stores past game states for eligibility trace calculations.
    /// Maintains a fixed-capacity buffer of feature vectors for temporal difference learning.
    /// </summary>
    class PastStatesBuffer(int capacity, NTupleManager nTupleManager) : IEnumerable<FeatureVector>
    {
        /// <summary>
        /// Gets the maximum capacity of the buffer.
        /// </summary>
        public int Capacity => _featureVecs.Length;
        
        /// <summary>
        /// Gets the current number of stored feature vectors.
        /// </summary>
        public int Count { get; private set; } = 0;

        readonly FeatureVector[] _featureVecs = [.. Enumerable.Range(0, capacity).Select(_ => new FeatureVector(nTupleManager))];
        int _loc = 0;

        /// <summary>
        /// Clears the buffer by resetting the location pointer.
        /// </summary>
        public void Clear() => _loc = 0;

        /// <summary>
        /// Adds a new feature vector to the buffer, overwriting the oldest entry if at capacity.
        /// </summary>
        /// <param name="featureVec">The feature vector to add</param>
        public void Add(FeatureVector featureVec)
        {
            featureVec.CopyTo(_featureVecs[_loc]);
            _loc = (_loc + 1) % Capacity;
            Count = Math.Min(Count + 1, Capacity);
        }

        /// <summary>
        /// Returns an enumerator that iterates through the buffer in reverse chronological order.
        /// </summary>
        /// <returns>An enumerator for the feature vectors</returns>
        public IEnumerator<FeatureVector> GetEnumerator() => new Enumerator(this);

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        /// <summary>
        /// Enumerator that provides reverse chronological iteration through the past states buffer.
        /// </summary>
        public class Enumerator : IEnumerator<FeatureVector>
        {
            /// <summary>
            /// Gets the current feature vector in the enumeration.
            /// </summary>
            public FeatureVector Current { get; private set; }

            object IEnumerator.Current => Current;

            readonly PastStatesBuffer _pastStatesBuffer;
            int _idx;
            int _moveCount;

            /// <summary>
            /// Initializes a new enumerator for the specified past states buffer.
            /// </summary>
            /// <param name="pastStatesBuffer">The buffer to enumerate</param>
            public Enumerator(PastStatesBuffer pastStatesBuffer)
            {
                _pastStatesBuffer = pastStatesBuffer;
                Reset();
                Debug.Assert(Current is not null);
            }

            /// <summary>
            /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
            /// </summary>
            public void Dispose() => GC.SuppressFinalize(this);

            /// <summary>
            /// Advances the enumerator to the next element in the buffer.
            /// </summary>
            /// <returns>true if the enumerator was successfully advanced; false if at the end</returns>
            public bool MoveNext()
            {
                if (_moveCount == _pastStatesBuffer.Count)
                    return false;

                var nextIdx = _idx - 1;
                if (nextIdx < 0)
                    nextIdx = _pastStatesBuffer.Count - 1;
                Current = _pastStatesBuffer._featureVecs[_idx];
                _idx = nextIdx;
                _moveCount++;
                return true;
            }

            /// <summary>
            /// Sets the enumerator to its initial position, which is before the first element.
            /// </summary>
            public void Reset()
            {
                Current = FeatureVector.Empty;
                _idx = _pastStatesBuffer._loc - 1;
                if (_idx < 0)
                    _idx = _pastStatesBuffer.Count - 1;
                _moveCount = 0;
            }
        }
    }
}