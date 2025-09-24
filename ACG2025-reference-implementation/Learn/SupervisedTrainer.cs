namespace ACG2025_reference_implementation.Learn;

using System;
using System.IO;
using System.Text;
using System.Linq;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

using ACG2025_reference_implementation.Utils;
using ACG2025_reference_implementation.Reversi;
using ACG2025_reference_implementation.Evaluation;

/// <summary>
/// A record class that defines configuration parameters for the supervised learning trainer.
/// Contains training epochs, learning rate, convergence criteria, file saving settings, etc.
/// </summary>
public record class SupervisedTrainerConfig
{
    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <value>Default is 200 epochs</value>
    public int NumEpoch { get; init; } = 200;
    
    /// <summary>
    /// Gets or sets the learning rate.
    /// </summary>
    /// <value>Default is 0.1</value>
    public double LearningRate { get; init; } = 0.1;
    
    /// <summary>
    /// Gets or sets the epsilon value used for convergence determination in numerical calculations.
    /// </summary>
    /// <value>Default is 1.0e-7</value>
    public double Epsilon { get; init; } = 1.0e-7;
    
    /// <summary>
    /// Gets or sets the weight of evaluation scores when combining evaluation scores with actual rewards.
    /// </summary>
    /// <value>Default is 0.5 (50% evaluation score, 50% actual reward)</value>
    public double EvalScoreFraction { get; init; } = 0.5;
    
    /// <summary>
    /// Gets or sets the patience value for early stopping.
    /// Training stops if the test loss worsens consecutively for this number of times.
    /// </summary>
    /// <value>Default is 0 (no early stopping)</value>
    public int Patience { get; init; } = 0;

    /// <summary>
    /// Gets or sets the base name of the weights file.
    /// </summary>
    /// <value>Default is "value_func_weights_sl"</value>
    public string WeightsFileName { get; init; } = "value_func_weights_sl";
    
    /// <summary>
    /// Gets or sets the loss history file name.
    /// </summary>
    /// <value>Default is "loss_history"</value>
    public string LossHistoryFileName { get; init; } = "loss_history";
    
    /// <summary>
    /// Gets or sets the interval (number of epochs) for saving weights.
    /// </summary>
    /// <value>Default is every 10 epochs</value>
    public int SaveWeightsInterval { get; init; } = 10;
    
    /// <summary>
    /// Gets or sets whether to save only the latest weights.
    /// If false, weights are saved with different filenames at each save interval.
    /// </summary>
    /// <value>Default is true (save only latest weights)</value>
    public bool SaveOnlyLatestWeights { get; init; } = true;
}

/// <summary>
/// A class for training Reversi value functions using supervised learning.
/// Uses the AdaGrad optimization algorithm with binary cross-entropy loss.
/// Provides efficient gradient calculation through parallel processing and early stopping functionality.
/// </summary>
/// <typeparam name="WeightType">The floating-point type used for weights (Half, float, double)</typeparam>
internal class SupervisedTrainer<WeightType> where WeightType : unmanaged, IFloatingPointIeee754<WeightType>
{
    /// <summary>
    /// Gets the trainer's label. Used for log output identification.
    /// </summary>
    public string Label { get; }

    /// <summary>Training configuration parameters</summary>
    readonly SupervisedTrainerConfig _config;
    /// <summary>Weights file path string format</summary>
    readonly string _weightsFilePath;
    /// <summary>Loss history file path</summary>
    readonly string _lossHistoryFilePath;
    /// <summary>StreamWriter for log output</summary>
    readonly StreamWriter _logger;

    /// <summary>Feature vector array for each thread</summary>
    FeatureVector[] _featureVecs = [];
    /// <summary>The value function to be trained</summary>
    readonly ValueFunctionForTrain<WeightType> _valueFunc;
    /// <summary>Weight gradient array for each thread</summary>
    WeightType[][] _weightGrads = [];
    /// <summary>Weight gradient square sum for AdaGrad</summary>
    WeightType[] _weightGradSquareSums = [];
    /// <summary>Bias gradient array for each thread</summary>
    WeightType[][] _biasGrad = [];
    /// <summary>Bias gradient square sum for AdaGrad</summary>
    readonly WeightType[] _biasGradSquareSum;
    /// <summary>Previous training loss</summary>
    double _prevTrainLoss;
    /// <summary>Previous test loss</summary>
    double _prevTestLoss;
    /// <summary>List of loss history</summary>
    readonly List<(double trainLoss, double testLoss)> _lossHistory = [];
    /// <summary>Overfitting counter</summary>
    int overfittingCount;
    /// <summary>Parallel processing options</summary>
    ParallelOptions? _parallelOptions;
    /// <summary>Epsilon value for numerical calculations</summary>
    readonly WeightType _epsilon;

    /// <summary>
    /// Initializes a new instance of SupervisedTrainer that outputs logs to standard output.
    /// </summary>
    /// <param name="valueFunc">The value function to be trained</param>
    /// <param name="config">Training configuration parameters</param>
    public SupervisedTrainer(ValueFunctionForTrain<WeightType> valueFunc, SupervisedTrainerConfig config)
        : this(valueFunc, config, Console.OpenStandardOutput()) { }

    /// <summary>
    /// Initializes a new instance of SupervisedTrainer that outputs logs to the specified stream.
    /// </summary>
    /// <param name="valueFunc">The value function to be trained</param>
    /// <param name="config">Training configuration parameters</param>
    /// <param name="logStream">The stream for log output</param>
    public SupervisedTrainer(ValueFunctionForTrain<WeightType> valueFunc, SupervisedTrainerConfig config, Stream logStream)
        : this(string.Empty, valueFunc, config, logStream) { }

    /// <summary>
    /// Initializes a new instance of SupervisedTrainer with a label that outputs logs to standard output.
    /// </summary>
    /// <param name="label">The trainer's label</param>
    /// <param name="valueFunc">The value function to be trained</param>
    /// <param name="config">Training configuration parameters</param>
    public SupervisedTrainer(string label, ValueFunctionForTrain<WeightType> valueFunc, SupervisedTrainerConfig config)
        : this(label, valueFunc, config, Console.OpenStandardOutput()) { }

    /// <summary>
    /// Initializes a new instance of SupervisedTrainer with a label that outputs logs to the specified stream.
    /// </summary>
    /// <param name="label">The trainer's label</param>
    /// <param name="valueFunc">The value function to be trained</param>
    /// <param name="config">Training configuration parameters</param>
    /// <param name="logStream">The stream for log output</param>
    public SupervisedTrainer(string label, ValueFunctionForTrain<WeightType> valueFunc, SupervisedTrainerConfig config, Stream logStream)
    {
        Label = label;
        _config = config;
        _epsilon = WeightType.CreateChecked(_config.Epsilon);
        _weightsFilePath = $"{config.WeightsFileName}{"{0}"}.bin";
        _lossHistoryFilePath = $"{config.LossHistoryFileName}.txt";
        _valueFunc = valueFunc;
        _biasGradSquareSum = new WeightType[_valueFunc.NumPhases];
        _logger = new StreamWriter(logStream) { AutoFlush = false };
    }

    /// <summary>
    /// Executes supervised learning with the default number of threads (processor count).
    /// </summary>
    /// <param name="trainData">Training dataset</param>
    /// <param name="testData">Test dataset</param>
    /// <param name="initAdaGrad">Whether to initialize AdaGrad gradient square sums</param>
    /// <param name="saveWeights">Whether to save weights</param>
    /// <param name="saveLossHistory">Whether to save loss history</param>
    /// <returns>A tuple of final training loss and test loss</returns>
    public (double trainLoss, double testLoss) Train(GameDataset trainData, GameDataset testData, bool initAdaGrad = true, bool saveWeights = true, bool saveLossHistory = true)
        => Train(trainData, testData, Environment.ProcessorCount, initAdaGrad, saveWeights, saveLossHistory);

    /// <summary>
    /// Executes supervised learning with the specified number of threads.
    /// Uses AdaGrad optimization algorithm to minimize binary cross-entropy loss.
    /// </summary>
    /// <param name="trainData">Training dataset</param>
    /// <param name="testData">Test dataset</param>
    /// <param name="numThreads">Number of threads to use for parallel processing</param>
    /// <param name="initAdaGrad">Whether to initialize AdaGrad gradient square sums</param>
    /// <param name="saveWeights">Whether to save weights</param>
    /// <param name="saveLossHistory">Whether to save loss history</param>
    /// <returns>A tuple of final training loss and test loss</returns>
    public (double trainLoss, double testLoss) Train(GameDataset trainData, GameDataset testData, int numThreads, bool initAdaGrad = true, bool saveWeights = true, bool saveLossHistory = true)
    {
        _parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = numThreads };

        InitBuffers(numThreads);

        if (initAdaGrad)
        {
            Array.Clear(_weightGradSquareSums);
            Array.Clear(_biasGradSquareSum);
        }

        _prevTrainLoss = _prevTestLoss = double.PositiveInfinity;
        overfittingCount = 0;

        PrintLabel();
        _logger.WriteLine("Start training.\n");
        _logger.Flush();

        var continueFlag = true;
        for (var epoch = 0; epoch < _config.NumEpoch && continueFlag; epoch++)
        {
            PrintLabel();
            continueFlag = ExecuteOneEpoch(trainData, testData);
            _logger.WriteLine($"Epoch {epoch + 1} has done.\n");

            if ((epoch + 1) % _config.SaveWeightsInterval == 0)
            {
                if (saveWeights)
                    SaveWeights(epoch + 1);

                if (saveLossHistory)
                    SaveLossHistory();
            }

            _logger.Flush();
        }

        if (saveWeights)
            SaveWeights(_config.NumEpoch);

        if (saveLossHistory)
            SaveLossHistory();

        _valueFunc.CopyWeightsBlackToWhite();

        return _lossHistory[^1];
    }

    /// <summary>
    /// Outputs the label to the log if a label is set.
    /// </summary>
    void PrintLabel()
    {
        if (!string.IsNullOrEmpty(Label))
            _logger.WriteLine($"[{Label}]");
    }

    /// <summary>
    /// Initializes buffers for parallel processing (feature vectors, gradient arrays).
    /// </summary>
    /// <param name="numThreads">Number of threads</param>
    void InitBuffers(int numThreads)
    {
        _featureVecs = [.. Enumerable.Range(0, numThreads).Select(_ => new FeatureVector(_valueFunc.NTupleManager))];

        var weights = _valueFunc.Weights;
        _weightGrads = [.. Enumerable.Range(0, numThreads).Select(_ => new WeightType[weights.Length / 2])];
        _weightGradSquareSums = new WeightType[weights.Length / 2];

        _biasGrad = [.. Enumerable.Range(0, numThreads).Select(_ => new WeightType[_valueFunc.NumPhases])];
    }

    /// <summary>
    /// Calculates the update factor for the AdaGrad algorithm.
    /// </summary>
    /// <param name="x">Gradient square sum</param>
    /// <returns>The value of 1/sqrt(x + epsilon)</returns>
    WeightType CalcAdaGradFactor(WeightType x) => WeightType.One / WeightType.Sqrt(x + _epsilon);

    /// <summary>
    /// Executes learning for one epoch.
    /// Performs test loss calculation, gradient calculation, weight updates, and convergence determination.
    /// </summary>
    /// <param name="trainData">Training dataset</param>
    /// <param name="testData">Test dataset</param>
    /// <returns>Whether to continue learning</returns>
    bool ExecuteOneEpoch(GameDataset trainData, GameDataset testData)
    {
        for (var i = 0; i < _parallelOptions!.MaxDegreeOfParallelism; i++)
        {
            Array.Clear(_weightGrads[i]);
            Array.Clear(_biasGrad[i]);
        }

        var testLoss = CalculateLoss(testData);

        _logger.WriteLine($"test loss: {testLoss}");

        var (trainLoss, numSamples) = CalculateGradients(trainData);

        _logger.WriteLine($"train loss: {trainLoss}");

        _lossHistory.Add((trainLoss, testLoss));

        var trainLossDiff = trainLoss - _prevTrainLoss;
        var testLossDiff = testLoss - _prevTestLoss;
        if (testLossDiff > _config.Epsilon)
        {
            if (++overfittingCount > _config.Patience)
            {
                _logger.WriteLine("early stopping.");
                return false;
            }
        }
        else
        {
            overfittingCount = 0;
        }

        if (Math.Abs(trainLossDiff) < _config.Epsilon)
        {
            _logger.WriteLine("converged.");
            return false;
        }

        ApplyGradients(numSamples);

        _prevTestLoss = testLoss;
        _prevTrainLoss = trainLoss;

        return true;
    }

    /// <summary>
    /// Saves weights to a file with the specified epoch number.
    /// Copies black weights to white before saving.
    /// </summary>
    /// <param name="epoch">Epoch number</param>
    void SaveWeights(int epoch)
    {
        var weightsLabel = _config.SaveOnlyLatestWeights ? string.Empty : $"_{epoch}";
        var path = string.Format(_weightsFilePath, weightsLabel);
        _valueFunc.CopyWeightsBlackToWhite();
        _valueFunc.SaveToFile(path);
    }

    /// <summary>
    /// Saves the history of training and test losses to a text file.
    /// Outputs in array format on each line.
    /// </summary>
    void SaveLossHistory()
    {
        var trainLossSb = new StringBuilder("[");
        var testLossSb = new StringBuilder("[");
        foreach ((var trainLoss, var testLoss) in _lossHistory)
        {
            trainLossSb.Append(trainLoss).Append(", ");
            testLossSb.Append(testLoss).Append(", ");
        }

        // remove last ", "
        trainLossSb.Remove(trainLossSb.Length - 2, 2);
        testLossSb.Remove(testLossSb.Length - 2, 2);

        trainLossSb.Append(']');
        testLossSb.Append(']');

        using var sw = new StreamWriter(_lossHistoryFilePath);
        sw.WriteLine(trainLossSb.ToString());
        sw.WriteLine(testLossSb.ToString());
    }

    /// <summary>
    /// Calculates gradients in parallel for the entire dataset.
    /// Each thread processes a different portion of the data and aggregates the results.
    /// </summary>
    /// <param name="dataset">The dataset to calculate gradients for</param>
    /// <returns>A tuple of total loss value and sample count</returns>
    unsafe (double loss, int numSamples) CalculateGradients(GameDataset dataset)
    {
        var numThreads = _parallelOptions!.MaxDegreeOfParallelism;
        var numDataPerThread = dataset.Length / numThreads;
        var lossSum = 0.0;
        var countSum = 0;
        Parallel.For(0, numThreads, _parallelOptions!, threadID =>
        {
            (var loss, var count) = CalculateGradients(threadID, dataset.AsSpan(threadID * numDataPerThread, numDataPerThread));
            AtomicOperations.Add(ref lossSum, loss);
            Interlocked.Add(ref countSum, count);
        });

        (var loss, var count) = CalculateGradients(0, dataset.AsSpan(numThreads * numDataPerThread));
        lossSum += loss;
        countSum += count;

        return (lossSum / countSum, countSum);
    }

    /// <summary>
    /// Calculates gradients for a portion of the dataset with the specified thread ID.
    /// Computes binary cross-entropy loss gradients for each move in each game
    /// and accumulates them in thread-specific gradient buffers.
    /// </summary>
    /// <param name="threadID">Thread ID (index for gradient buffers)</param>
    /// <param name="dataset">The portion of the dataset to calculate gradients for</param>
    /// <returns>A tuple of loss value and sample count</returns>
    unsafe (double loss, int count) CalculateGradients(int threadID, Span<GameDatasetItem> dataset)
    {
        var loss = 0.0;
        var featureVec = _featureVecs[threadID];
        var frac = WeightType.CreateChecked(_config.EvalScoreFraction);
        var count = 0;

        fixed (int* nTupleOffset = _valueFunc.NTupleOffset)
        fixed (WeightType* wg = _weightGrads[threadID])
        {
            for (var i = 0; i < dataset.Length; i++)
            {
                ref var data = ref dataset[i];
                var pos = data.RootPos;

                featureVec.Init(ref pos);

                for (var j = 0; j < data.Moves.Length; j++)
                {
                    var reward = GetReward(ref data, pos.SideToMove);
                    var evalScore = (data.EvalScores.Length == data.Moves.Length) ? WeightType.CreateChecked(data.EvalScores[j]) : WeightType.NaN;
                    var value = _valueFunc.PredictWithBlackWeights(featureVec);
                    WeightType target;
                    WeightType delta;

                    if (WeightType.IsNaN(evalScore))
                        target = reward;
                    else
                        target = (WeightType.One - frac) * reward + frac * evalScore;

                    delta = value - target;

                    loss += double.CreateChecked(MathFunctions.BinaryCrossEntropy(value, target));
                    count++;

                    if (pos.SideToMove == DiscColor.Black)
                        CalcGrads<Black>(nTupleOffset, wg, delta);
                    else
                        CalcGrads<White>(nTupleOffset, wg, delta);

                    ref var nextMove = ref data.Moves[j];
                    if (nextMove.Coord != BoardCoordinate.Pass)
                    {
                        pos.Update(ref nextMove);
                        featureVec.Update(ref nextMove);
                    }
                    else
                    {
                        pos.Pass();
                        featureVec.Pass();
                    }
                }
            }
        }

        return (loss, count);

        /// <summary>
        /// Calculates weight and bias gradients for the specified side to move.
        /// Performs gradient updates considering symmetries for each feature in the N-tuple system.
        /// </summary>
        /// <typeparam name="DiscColor">The color of the side to move (Black or White)</typeparam>
        /// <param name="nTupleOffset">Pointer to the N-tuple offset array</param>
        /// <param name="weightGrads">Pointer to the weight gradient array</param>
        /// <param name="delta">Error (predicted value - target value)</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        unsafe void CalcGrads<DiscColor>(int* nTupleOffset, WeightType* weightGrads, WeightType delta) where DiscColor : struct, IDiscColor
        {
            var phase = _valueFunc.EmptyCellCountToPhase[featureVec.EmptyCellCount];
            for (var nTupleID = 0; nTupleID < featureVec.NTupleManager.NumNTuples; nTupleID++)
            {
                var offset = _valueFunc.PhaseOffset[phase] + nTupleOffset[nTupleID];
                var wg = weightGrads + offset;
                ref Feature feature = ref featureVec.GetFeature(nTupleID);
                fixed (int* opp = _valueFunc.NTupleManager.GetRawOpponentFeatureTable(nTupleID))
                fixed (int* mirror = featureVec.NTupleManager.GetRawMirroredFeatureTable(nTupleID))
                {
                    for (var k = 0; k < feature.Length; k++)
                    {
                        var f = (typeof(DiscColor) == typeof(Black)) ? feature[k] : opp[feature[k]];
                        var mf = mirror[f];

                        wg[f] += delta;

                        if (mf != f)
                            wg[mf] += delta;
                    }
                }
            }
            _biasGrad[threadID][phase] += delta;
        }
    }

    /// <summary>
    /// Applies calculated gradients to weights using the AdaGrad algorithm.
    /// Aggregates gradients from all threads, then updates weights and biases with the applied learning rate.
    /// </summary>
    /// <param name="numSamples">Number of samples (used for gradient normalization)</param>
    unsafe void ApplyGradients(int numSamples)
    {
        var numThreads = _parallelOptions!.MaxDegreeOfParallelism;
        var eta = WeightType.CreateChecked(_config.LearningRate / numSamples);

        fixed (WeightType* wg = _weightGrads[0])
        {
            for (var threadID = 1; threadID < numThreads; threadID++)
                for (var i = 0; i < _weightGrads[threadID].Length; i++)
                    wg[i] += _weightGrads[threadID][i];
        }

        fixed (WeightType* bg = _biasGrad[0])
        {
            for (var threadID = 1; threadID < numThreads; threadID++)
                for (var i = 0; i < _biasGrad[threadID].Length; i++)
                    bg[i] += _biasGrad[threadID][i];
        }

        fixed (WeightType* w = _valueFunc.Weights)
        fixed (WeightType* wg = _weightGrads[0])
        fixed (WeightType* wg2 = _weightGradSquareSums)
        {
            for (var i = 0; i < _valueFunc.Weights.Length / 2; i++)
            {
                var g = wg[i];
                wg2[i] += g * g;
                w[i] -= eta * CalcAdaGradFactor(wg2[i]) * g;
            }
        }

        fixed (WeightType* bg = _biasGrad[0])
        fixed (WeightType* bg2 = _biasGradSquareSum)
        {
            for (var i = 0; i < _valueFunc.Bias.Length; i++)
            {
                bg2[i] += bg[i] * bg[i];
                _valueFunc.Bias[i] -= eta * CalcAdaGradFactor(bg2[i]) * bg[i];
            }
        }
    }

    /// <summary>
    /// Calculates binary cross-entropy loss for the dataset.
    /// Only computes loss values without gradient calculation.
    /// </summary>
    /// <param name="dataset">The dataset to evaluate</param>
    /// <returns>Average loss value</returns>
    double CalculateLoss(GameDataset dataset)
    {
        var loss = 0.0;
        var count = 0;
        var frac = WeightType.CreateChecked(_config.EvalScoreFraction);
        var featureVec = _featureVecs[0];
        for (var i = 0; i < dataset.Length; i++)
        {
            ref var data = ref dataset[i];
            var pos = data.RootPos;

            featureVec.Init(ref pos);

            for (var j = 0; j < data.Moves.Length; j++)
            {
                var reward = GetReward(ref data, pos.SideToMove);
                var evalScore = (data.EvalScores.Length == data.Moves.Length) ? WeightType.CreateChecked(data.EvalScores[j]) : WeightType.NaN;

                WeightType target;
                if (WeightType.IsNaN(evalScore))
                    target = reward;
                else
                    target = (WeightType.One - frac) * reward + frac * evalScore;

                loss += double.CreateChecked(MathFunctions.BinaryCrossEntropy(_valueFunc.PredictWithBlackWeights(featureVec), target));
                count++;

                ref var move = ref data.Moves[j];
                if (move.Coord != BoardCoordinate.Pass)
                {
                    pos.Update(ref move);
                    featureVec.Update(ref move);
                }
                else
                {
                    pos.Pass();
                    featureVec.Pass();
                }
            }
        }
        return loss / count;
    }

    /// <summary>
    /// Gets the reward value for the specified side to move from the game result.
    /// Converts victory to 1.0, defeat to 0.0, and draw to 0.5.
    /// </summary>
    /// <param name="data">Game dataset item</param>
    /// <param name="sideToMove">The side to move for which to calculate the reward</param>
    /// <returns>Reward value (0.0, 0.5, or 1.0)</returns>
    static WeightType GetReward(ref GameDatasetItem data, DiscColor sideToMove)
    {
        var score = data.ScoreFromBlack;

        if (sideToMove != DiscColor.Black)
            score *= -1;

        if (score == 0)
            return WeightType.CreateChecked(0.5);

        return (score > 0) ? WeightType.One : WeightType.Zero;
    }
}