global using QWeightType = System.Int16;

namespace ACG2025_reference_implementation.Evaluation;

using System;
using System.IO;
using System.Numerics;
using System.Text;
using System.Linq;
using System.Runtime.CompilerServices;

using ACG2025_reference_implementation.Utils;
using ACG2025_reference_implementation.Reversi;
using ACG2025_reference_implementation.NTupleSystem;

using static ACG2025_reference_implementation.Reversi.Constants;
using static ACG2025_reference_implementation.Evaluation.ValueFunctionConstantConfig;

/// <summary>
/// A value function optimized for search operations using quantized 16-bit integer weights.
/// This class is designed for high-performance evaluation during alpha-beta search, where integer arithmetic
/// provides better compatibility with null window searches and avoids floating-point precision issues.
/// The weights are quantized from trained floating-point values to enable fast integer-only computation.
/// </summary>
internal class ValueFunction
{
    /// <summary>
    /// The minimum value that can be returned by the value function.
    /// </summary>
    public const int ValueMin = -30000;
    
    /// <summary>
    /// The maximum value that can be returned by the value function.
    /// </summary>
    public const int ValueMax = 30000;
    
    /// <summary>
    /// An invalid value used to indicate error conditions or uninitialized states.
    /// </summary>
    public const int ValueInvalid = ValueMin - 1;

    /// <summary>
    /// Output scaling factor used to convert internal values to the final output range.
    /// </summary>
    public const int OutScale = 600;
    
    /// <summary>
    /// Internal scaling factor used during weight quantization to preserve precision.
    /// Applied during quantization and removed during evaluation to maintain accuracy in integer arithmetic.
    /// </summary>
    public const int FVScale = 32;

    /// <summary>
    /// Gets the number of game phases used for evaluation.
    /// </summary>
    public int NumPhases { get; }
    
    /// <summary>
    /// Gets the number of moves per phase.
    /// </summary>
    public int NumMovesPerPhase { get; }
    
    /// <summary>
    /// Gets the mapping from empty cell count to phase index.
    /// </summary>
    public ReadOnlySpan<int> EmptySquareCountToPhase => _emptyCountToPhase;

    /// <summary>
    /// Gets the n-tuple manager that defines the n-tuples used for feature extraction.
    /// </summary>
    public NTupleManager NTupleManager { get; }

    /// <summary>
    /// Gets the quantized weight array containing all weights for all phases and colors.
    /// </summary>
    public QWeightType[] Weights { get; private set; }
    
    /// <summary>
    /// Gets the quantized bias values for each phase.
    /// </summary>
    public QWeightType[] Bias { get; private set; }
    
    /// <summary>
    /// Gets the offset for each disc color in the weight array.
    /// </summary>
    public ReadOnlySpan<int> DiscColorOffset => _discColorOffset;
    
    /// <summary>
    /// Gets the offset for each phase in the weight array.
    /// </summary>
    public ReadOnlySpan<int> PhaseOffset => _phaseOffset;
    
    /// <summary>
    /// Gets the offset for each n-tuple in the weight array.
    /// </summary>
    public ReadOnlySpan<int> NTupleOffset => _nTupleOffset;

    readonly int[] _emptyCountToPhase;
    readonly int[] _discColorOffset;
    readonly int[] _phaseOffset;
    readonly int[] _nTupleOffset;

    /// <summary>
    /// Initializes a new instance of ValueFunction with the specified n-tuple manager and moves per phase.
    /// </summary>
    /// <param name="nTupleManager">The n-tuple manager that defines the n-tuples</param>
    /// <param name="numMovesPerPhase">The number of moves per phase</param>
    public ValueFunction(NTupleManager nTupleManager, int numMovesPerPhase)
    {
        NumMovesPerPhase = numMovesPerPhase;
        NumPhases = (NumCells - 4) / numMovesPerPhase;
        _emptyCountToPhase = new int[NumCells];
        InitEmptyCountToPhaseTable();

        NTupleManager = nTupleManager;
        var numFeatures = nTupleManager.NumFeatures.Sum();
        Weights = new QWeightType[2 * NumPhases * numFeatures];
        _discColorOffset = [0, Weights.Length / 2];
        Bias = new QWeightType[NumPhases];

        _phaseOffset = new int[NumPhases];
        for (var i = 0; i < _phaseOffset.Length; i++)
            _phaseOffset[i] = i * numFeatures;

        _nTupleOffset = new int[nTupleManager.NumNTuples];
        _nTupleOffset[0] = 0;
        for (var i = 1; i < _nTupleOffset.Length; i++)
            _nTupleOffset[i] += _nTupleOffset[i - 1] + nTupleManager.NumFeatures[i - 1];
    }

    void InitEmptyCountToPhaseTable()
    {
        for (var phase = 0; phase < NumPhases; phase++)
        {
            var offset = phase * NumMovesPerPhase;
            for (var i = 0; i < NumMovesPerPhase; i++)
                _emptyCountToPhase[NumCells - 4 - offset - i] = phase;
        }
        _emptyCountToPhase[0] = NumPhases - 1;
    }

    /// <summary>
    /// Loads a value function from a binary file and converts it to quantized integer format.
    /// </summary>
    /// <param name="path">Path to the file to load</param>
    /// <returns>A new ValueFunction instance loaded from the file</returns>
    /// <exception cref="InvalidDataException">Thrown when the file format is invalid</exception>
    /// <remarks>
    /// File format:
    /// - offset 0: label (for endianness check)
    /// - offset 10: the number of N-Tuples
    /// - offset 14: N-Tuple's coordinates
    /// - offset M: the size of weight
    /// - offset M + 4: the number of moves per phase
    /// - offset M + 8: weights
    /// - offset M + 8 + NumWeights: bias
    /// </remarks>
    public static ValueFunction LoadFromFile(string path)
    {
        const int BufferSize = 16;

        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
        Span<byte> buffer = stackalloc byte[BufferSize];
        fs.Read(buffer[..LabelSize]);
        var label = Encoding.ASCII.GetString(buffer[..LabelSize]);
        var swapBytes = label == LabelReversed;

        if (!swapBytes && label != Label)
            throw new InvalidDataException($"The format of \"{path}\" is invalid.");

        // load N-Tuples
        fs.Read(buffer[..sizeof(int)], swapBytes);
        var numNTuples = BitConverter.ToInt32(buffer);
        var nTuples = new NTuple[numNTuples];
        for (var i = 0; i < nTuples.Length; i++)
        {
            fs.Read(buffer[..sizeof(int)], swapBytes);
            var size = BitConverter.ToInt32(buffer);
            var coords = new BoardCoordinate[size];
            for (var j = 0; j < size; j++)
                coords[j] = (BoardCoordinate)fs.ReadByte();
            nTuples[i] = new NTuple(coords);
        }

        // load weights
        fs.Read(buffer[..sizeof(int)], swapBytes);
        var weightSize = BitConverter.ToInt32(buffer);
        if (weightSize != 2 && weightSize != 4 && weightSize != 8)
            throw new InvalidDataException($"The size {weightSize} is invalid for weight.");

        fs.Read(buffer[..sizeof(int)], swapBytes);
        var numMovesPerPhase = BitConverter.ToInt32(buffer);
        var valueFunc = new ValueFunction(new NTupleManager(nTuples), numMovesPerPhase);
        var numPhases = valueFunc.NumPhases;

        var packedWeights = Enumerable.Range(0, numPhases).Select(p => new double[nTuples.Length][]).ToArray();
        for (var phase = 0; phase < packedWeights.Length; phase++)
        {
            for (var nTupleID = 0; nTupleID < packedWeights[phase].Length; nTupleID++)
            {
                fs.Read(buffer[..sizeof(int)], swapBytes);
                var size = BitConverter.ToInt32(buffer);
                var pw = packedWeights[phase][nTupleID] = new double[size];
                for (var i = 0; i < pw.Length; i++)
                {
                    fs.Read(buffer[..weightSize], swapBytes);
                    if (weightSize == 2)
                        pw[i] = (double)BitConverter.ToHalf(buffer);
                    else if (weightSize == 4)
                        pw[i] = BitConverter.ToSingle(buffer);
                    else if (weightSize == 8)
                        pw[i] = BitConverter.ToDouble(buffer);
                }
            }
        }

        Span<double> bias = stackalloc double[numPhases];
        for (var phase = 0; phase < numPhases; phase++)
        {
            fs.Read(buffer[..weightSize], swapBytes);
            if (weightSize == 2)
                bias[phase] = (double)BitConverter.ToHalf(buffer);
            else if (weightSize == 4)
                bias[phase] = BitConverter.ToSingle(buffer);
            else if (weightSize == 8)
                bias[phase] = BitConverter.ToDouble(buffer);
        }

        for (var phase = 0; phase < numPhases; phase++)
            valueFunc.Bias[phase] = (QWeightType)Math.Clamp((int)MathFunctions.Round(bias[phase] * OutScale * FVScale), QWeightType.MinValue, QWeightType.MaxValue);

        // expand weights
        valueFunc.Weights = valueFunc.ExpandAndQuantizePackedWeights(packedWeights);
        valueFunc.CopyWeightsBlackToWhite();

        return valueFunc;
    }

    /// <summary>
    /// Creates a quantized ValueFunction from a trained floating-point ValueFunctionForTrain.
    /// Converts floating-point weights to 16-bit integers with appropriate scaling.
    /// </summary>
    /// <typeparam name="WeightType">The floating-point type used in the source value function</typeparam>
    /// <param name="srcVf">The source trained value function to convert</param>
    /// <returns>A new quantized ValueFunction instance</returns>
    public static ValueFunction CreateFromTrainedValueFunction<WeightType>(ValueFunctionForTrain<WeightType> srcVf) where WeightType : unmanaged, IFloatingPointIeee754<WeightType>
    {
        var valueFunc = new ValueFunction(srcVf.NTupleManager, srcVf.NumMovesPerPhase);
        for (var i = 0; i < valueFunc.Weights.Length; i++)
            valueFunc.Weights[i] = QWeightType.CreateChecked(MathFunctions.Round(srcVf.Weights[i] * WeightType.CreateChecked(OutScale * FVScale)));

        for (var i = 0; i < valueFunc.Bias.Length; i++)
            valueFunc.Bias[i] = QWeightType.CreateChecked(MathFunctions.Round(srcVf.Bias[i] * WeightType.CreateChecked(OutScale * FVScale)));

        return valueFunc;
    }

    /// <summary>
    /// Evaluates the position using quantized integer arithmetic for high-speed search operations.
    /// Returns a scaled integer value suitable for alpha-beta search with null window optimizations.
    /// </summary>
    /// <param name="featureVec">The feature vector representing the current position</param>
    /// <returns>An integer evaluation value clamped between ValueMin and ValueMax</returns>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public unsafe int F(FeatureVector featureVec)
    {
        int phase;
        fixed (int* toPhase = EmptySquareCountToPhase)
            phase = toPhase[featureVec.EmptyCellCount];

        var y = 0;
        fixed (int* discColorOffset = this._discColorOffset)
        fixed (QWeightType* weights = &Weights[this._discColorOffset[(int)featureVec.SideToMove] + _phaseOffset[phase]])
        fixed (Feature* features = featureVec.Features)
        {
            for (var nTupleID = 0; nTupleID < _nTupleOffset.Length; nTupleID++)
            {
                var w = weights + _nTupleOffset[nTupleID];
                ref Feature feature = ref features[nTupleID];
                for (var i = 0; i < feature.Length; i++)
                    y += w[feature[i]];
            }

            fixed (QWeightType* bias = Bias)
                y += bias[phase];
        }

        return Math.Min(Math.Max(y / FVScale, ValueMin), ValueMax);
    }

    /// <summary>
    /// Predicts the win rate as a floating-point probability by converting the integer evaluation
    /// to a logit value and applying the sigmoid function.
    /// </summary>
    /// <typeparam name="T">The floating-point type for the result</typeparam>
    /// <param name="featureVec">The feature vector representing the current position</param>
    /// <returns>A probability value between 0 and 1 representing the win rate</returns>
    public T PredictWinRate<T>(FeatureVector featureVec) where T : struct, IFloatingPointIeee754<T>
    {
        var logit = T.CreateChecked(F(featureVec)) / T.CreateChecked(OutScale);
        return MathFunctions.StdSigmoid(logit);
    }

    /// <summary>
    /// Expands packed weights and quantizes them to 16-bit integers.
    /// Converts floating-point weights from the packed format to the expanded quantized format
    /// used for fast evaluation.
    /// </summary>
    /// <param name="packedWeights">The packed floating-point weights to expand and quantize</param>
    /// <returns>A flat array of quantized weights</returns>
    QWeightType[] ExpandAndQuantizePackedWeights(double[][][] packedWeights)
    {
        var numPhases = packedWeights.Length;
        var qweights = new QWeightType[2 * numPhases * NTupleManager.NumFeatures.Sum()];

        for (var phase = 0; phase < numPhases; phase++)
        {
            for (var nTupleID = 0; nTupleID < _nTupleOffset.Length; nTupleID++)
            {
                var qw = qweights.AsSpan(_phaseOffset[phase] + _nTupleOffset[nTupleID], NTupleManager.NumFeatures[nTupleID]);
                var pw = packedWeights[phase][nTupleID];
                ReadOnlySpan<int> mirror = NTupleManager.GetMirroredFeatureTable(nTupleID);
                var i = 0;
                for (var feature = 0; feature < qw.Length; feature++)
                {
                    var mirrored = mirror[feature];
                    if (feature <= mirrored)
                        qw[feature] = (QWeightType)Math.Clamp((int)MathFunctions.Round(pw[i++] * OutScale * FVScale), QWeightType.MinValue, QWeightType.MaxValue);
                    else
                        qw[feature] = qw[mirrored];
                }
            }
        }
        return qweights;
    }

    /// <summary>
    /// Copies quantized weights from black to white by applying opponent feature transformation.
    /// Ensures that white's weights are properly set up for evaluation from white's perspective.
    /// </summary>
    void CopyWeightsBlackToWhite()
    {
        var whiteOffset = _discColorOffset[(int)DiscColor.White];
        Span<QWeightType> bWeights = Weights.AsSpan(0, whiteOffset);
        Span<QWeightType> wWeights = Weights.AsSpan(whiteOffset);

        for (var phase = 0; phase < NumPhases; phase++)
        {
            for (var nTupleID = 0; nTupleID < _nTupleOffset.Length; nTupleID++)
            {
                var bw = bWeights[(PhaseOffset[phase] + _nTupleOffset[nTupleID])..];
                var ww = wWeights[(PhaseOffset[phase] + _nTupleOffset[nTupleID])..];
                ReadOnlySpan<int> toOpponent = NTupleManager.GetOpponentFeatureTable(nTupleID);
                for (var feature = 0; feature < toOpponent.Length; feature++)
                    ww[feature] = bw[toOpponent[feature]];
            }
        }
    }
}