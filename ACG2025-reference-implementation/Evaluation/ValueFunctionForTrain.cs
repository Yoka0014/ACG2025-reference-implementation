using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.IO;
using System.Text;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using ACG2025_reference_implementation.Utils;
using ACG2025_reference_implementation.Reversi;
using ACG2025_reference_implementation.NTupleSystem;

using static ACG2025_reference_implementation.Reversi.Constants;
using static ACG2025_reference_implementation.Evaluation.ValueFunctionConstantConfig;

namespace ACG2025_reference_implementation.Evaluation;

/// <summary>
/// A value function for training using floating-point weights.
/// Supports phase-based evaluation where different weights are used for different game phases
/// based on the number of empty cells remaining on the board.
/// </summary>
/// <typeparam name="WeightType">The floating-point type used for weights (Half, float, or double)</typeparam>
internal class ValueFunctionForTrain<WeightType> where WeightType : unmanaged, IFloatingPointIeee754<WeightType>
{
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
    public ReadOnlySpan<int> EmptyCellCountToPhase => _emptyCellCountToPhase;

    /// <summary>
    /// Gets the n-tuple manager that defines the n-tuples used for feature extraction.
    /// </summary>
    public NTupleManager NTupleManager { get; }

    /// <summary>
    /// Gets or sets the weight array containing all weights for all phases and colors.
    /// </summary>
    public WeightType[] Weights { get; private set; }
    
    /// <summary>
    /// Gets or sets the bias values for each phase.
    /// </summary>
    public WeightType[] Bias { get; private set; }
    
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

    readonly int[] _emptyCellCountToPhase;
    readonly int[] _discColorOffset;
    readonly int[] _phaseOffset;
    readonly int[] _nTupleOffset;

    /// <summary>
    /// Initializes a new instance of ValueFunctionForTrain with the specified n-tuple manager and moves per phase.
    /// </summary>
    /// <param name="nTupleManager">The n-tuple manager that defines the n-tuples</param>
    /// <param name="numMovesPerPhase">The number of moves per phase</param>
    public ValueFunctionForTrain(NTupleManager nTupleManager, int numMovesPerPhase = NumInitialCells)
    {
        NumMovesPerPhase = numMovesPerPhase;
        NumPhases = (NumCells - 4) / numMovesPerPhase;
        _emptyCellCountToPhase = new int[NumCells];
        InitEmptyCountToPhaseTable();

        NTupleManager = nTupleManager;
        var numFeatures = nTupleManager.NumFeatures.Sum();
        Weights = new WeightType[2 * NumPhases * numFeatures];
        _discColorOffset = [0, Weights.Length / 2];
        Bias = new WeightType[NumPhases];

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
                _emptyCellCountToPhase[NumCells - 4 - offset - i] = phase;
        }
        _emptyCellCountToPhase[0] = NumPhases - 1;
    }

    /// <summary>
    /// Loads a value function from a binary file.
    /// </summary>
    /// <param name="filePath">Path to the file to load</param>
    /// <returns>A new ValueFunctionForTrain instance loaded from the file</returns>
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
    public static ValueFunctionForTrain<WeightType> LoadFromFile(string filePath)
    {
        const int BufferSize = 16;

        using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read);
        Span<byte> buffer = stackalloc byte[BufferSize];
        fs.Read(buffer[..LabelSize]);
        var label = Encoding.ASCII.GetString(buffer[..LabelSize]);
        var swapBytes = label == LabelReversed;

        if (!swapBytes && label != Label)
            throw new InvalidDataException($"The format of \"{filePath}\" is invalid.");

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
        var valueFunc = new ValueFunctionForTrain<WeightType>(new NTupleManager(nTuples), numMovesPerPhase);
        var numPhases = valueFunc.NumPhases;

        var packedWeights = Enumerable.Range(0, numPhases).Select(p => new WeightType[nTuples.Length][]).ToArray();
        for (var phase = 0; phase < packedWeights.Length; phase++)
        {
            for (var nTupleID = 0; nTupleID < packedWeights[phase].Length; nTupleID++)
            {
                fs.Read(buffer[..sizeof(int)], swapBytes);
                var size = BitConverter.ToInt32(buffer);
                var pw = packedWeights[phase][nTupleID] = new WeightType[size];
                for (var i = 0; i < pw.Length; i++)
                {
                    fs.Read(buffer[..weightSize], swapBytes);
                    if (weightSize == 2)
                        pw[i] = WeightType.CreateChecked(BitConverter.ToHalf(buffer));
                    else if (weightSize == 4)
                        pw[i] = WeightType.CreateChecked(BitConverter.ToSingle(buffer));
                    else if (weightSize == 8)
                        pw[i] = WeightType.CreateChecked(BitConverter.ToDouble(buffer));
                }
            }
        }

        for (var phase = 0; phase < numPhases; phase++)
        {
            fs.Read(buffer[..weightSize], swapBytes);
            if (weightSize == 2)
                valueFunc.Bias[phase] = WeightType.CreateChecked(BitConverter.ToHalf(buffer));
            else if (weightSize == 4)
                valueFunc.Bias[phase] = WeightType.CreateChecked(BitConverter.ToSingle(buffer));
            else if (weightSize == 8)
                valueFunc.Bias[phase] = WeightType.CreateChecked(BitConverter.ToDouble(buffer));
        }

        // expand weights
        valueFunc.Weights = valueFunc.ExpandPackedWeights(packedWeights);
        valueFunc.CopyWeightsBlackToWhite();

        return valueFunc;
    }

    /// <summary>
    /// Gets the weights for the specified color and phase.
    /// </summary>
    /// <param name="color">The disc color</param>
    /// <param name="phase">The game phase</param>
    /// <returns>A span of weights for the specified color and phase</returns>
    public Span<WeightType> GetWeights(DiscColor color, int phase)
        => Weights.AsSpan(_discColorOffset[(int)color])[PhaseOffset[phase]..PhaseOffset[phase + 1]];

    public void InitWithZeros()
    {
        Array.Fill(Weights, WeightType.Zero);
        Array.Fill(Bias, WeightType.Zero);
    }

    /// <summary>
    /// Copies weights from black to white by applying opponent feature transformation.
    /// </summary>
    public void CopyWeightsBlackToWhite()
    {
        var whiteOffset = _discColorOffset[(int)DiscColor.White];
        Span<WeightType> bWeights = Weights.AsSpan(0, whiteOffset);
        Span<WeightType> wWeights = Weights.AsSpan(whiteOffset);

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

    /// <summary>
    /// Predicts the winning logit value for the given feature vector using automatic phase detection.
    /// </summary>
    /// <param name="featureVector">The feature vector to evaluate</param>
    /// <returns>The predicted logit value</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public unsafe WeightType PredictLogit(FeatureVector featureVector)
    {
        int phase;
        fixed (int* toPhase = EmptyCellCountToPhase)
            phase = toPhase[featureVector.EmptyCellCount];
        return PredictLogit(featureVector, phase);
    }

    /// <summary>
    /// Predicts the winning logit value for the given feature vector and specified phase.
    /// </summary>
    /// <param name="featureVector">The feature vector to evaluate</param>
    /// <param name="phase">The game phase to use for evaluation</param>
    /// <returns>The predicted logit value</returns>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public unsafe WeightType PredictLogit(FeatureVector featureVector, int phase)
    {
        Debug.Assert(phase < NumPhases);

        var x = WeightType.Zero;
        fixed (int* discColorOffset = _discColorOffset)
        fixed (WeightType* weights = &Weights[_discColorOffset[(int)featureVector.SideToMove] + _phaseOffset[phase]])
        fixed (Feature* features = featureVector.Features)
        {
            for (var nTupleID = 0; nTupleID < _nTupleOffset.Length; nTupleID++)
            {
                var w = weights + _nTupleOffset[nTupleID];
                ref Feature feature = ref features[nTupleID];
                for (var i = 0; i < feature.Length; i++)
                    x += w[feature[i]];
            }

            fixed (WeightType* bias = Bias)
                x += bias[phase];
        }

        return x;
    }

    /// <summary>
    /// Predicts the winning logit value using black weights.
    /// </summary>
    /// <param name="featureVector">The feature vector to evaluate</param>
    /// <returns>The predicted logit value using black weights</returns>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public unsafe WeightType PredictLogitWithBlackWeights(FeatureVector featureVector)
    {
        int phase;
        fixed (int* toPhase = EmptyCellCountToPhase)
            phase = toPhase[featureVector.EmptyCellCount];
        return PredictLogitWithBlackWeights(featureVector, phase);
    }

    /// <summary>
    /// Predicts the winning logit value using black weights for the specified phase.
    /// </summary>
    /// <param name="posFeatureVec">The feature vector to evaluate</param>
    /// <param name="phase">The game phase to use for evaluation</param>
    /// <returns>The predicted logit value using black weights</returns>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public unsafe WeightType PredictLogitWithBlackWeights(FeatureVector posFeatureVec, int phase)
    {
        if (posFeatureVec.SideToMove == DiscColor.Black)
            return PredictLogit(posFeatureVec, phase);

        Debug.Assert(phase < NumPhases);

        var x = WeightType.Zero;
        fixed (int* discColorOffset = _discColorOffset)
        fixed (WeightType* weights = &Weights[_discColorOffset[(int)DiscColor.Black] + _phaseOffset[phase]])
        fixed (Feature* features = posFeatureVec.Features)
        {
            for (var nTupleID = 0; nTupleID < _nTupleOffset.Length; nTupleID++)
            {
                var w = weights + _nTupleOffset[nTupleID];
                ref Feature feature = ref features[nTupleID];
                fixed (int* toOpp = NTupleManager.GetRawOpponentFeatureTable(nTupleID))
                {
                    for (var i = 0; i < feature.Length; i++)
                        x += w[toOpp[feature[i]]];
                }
            }

            fixed (WeightType* bias = Bias)
                x += bias[phase];
        }

        return x;
    }

    /// <summary>
    /// Predicts the winning probability by applying sigmoid to the logit.
    /// </summary>
    /// <param name="pfv">The feature vector to evaluate</param>
    /// <returns>The predicted probability value</returns>
    public WeightType Predict(FeatureVector pfv) => MathFunctions.StdSigmoid(PredictLogit(pfv));
    
    /// <summary>
    /// Predicts the winning probability by applying sigmoid to the logit for the specified phase.
    /// </summary>
    /// <param name="pfv">The feature vector to evaluate</param>
    /// <param name="phase">The game phase to use for evaluation</param>
    /// <returns>The predicted probability value</returns>
    public WeightType Predict(FeatureVector pfv, int phase) => MathFunctions.StdSigmoid(PredictLogit(pfv, phase));

    /// <summary>
    /// Predicts the winning probability using black weights.
    /// </summary>
    /// <param name="pfv">The feature vector to evaluate</param>
    /// <returns>The predicted probability value using black weights</returns>
    public WeightType PredictWithBlackWeights(FeatureVector pfv)
        => (pfv.SideToMove == DiscColor.Black) ? Predict(pfv) : MathFunctions.StdSigmoid(PredictLogitWithBlackWeights(pfv));

    /// <summary>
    /// Predicts the winning probability using black weights for the specified phase.
    /// </summary>
    /// <param name="pfv">The feature vector to evaluate</param>
    /// <param name="phase">The game phase to use for evaluation</param>
    /// <returns>The predicted probability value using black weights</returns>
    public WeightType PredictWithBlackWeights(FeatureVector pfv, int phase)
        => (pfv.SideToMove == DiscColor.Black) ? Predict(pfv, phase) : MathFunctions.StdSigmoid(PredictLogitWithBlackWeights(pfv, phase));

    /// <summary>
    /// Saves the value function to a binary file.
    /// </summary>
    /// <param name="filePath">Path to the file to save</param>
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
    public void SaveToFile(string filePath)
    {
        using var fs = new FileStream(filePath, FileMode.OpenOrCreate, FileAccess.Write);
        fs.Write(Encoding.ASCII.GetBytes(Label), 0, LabelSize);

        // save N-Tuples
        ReadOnlySpan<NTuple> nTuples = NTupleManager.NTuples;
        fs.Write(BitConverter.GetBytes(nTuples.Length));
        for (var nTupleID = 0; nTupleID < nTuples.Length; nTupleID++)
        {
            var coords = nTuples[nTupleID].GetCoordinates(0);
            fs.Write(BitConverter.GetBytes(coords.Length));
            foreach (var coord in coords)
                fs.WriteByte((byte)coord);
        }

        // save weights
        var packedWeights = PackWeights();
        var weightSize = Marshal.SizeOf<WeightType>();
        fs.Write(BitConverter.GetBytes(weightSize));
        Span<byte> weightBytes = stackalloc byte[weightSize];

        fs.Write(BitConverter.GetBytes(NumMovesPerPhase));

        for (var phase = 0; phase < packedWeights.Length; phase++)
        {
            for (var nTupleID = 0; nTupleID < packedWeights[phase].Length; nTupleID++)
            {
                var pw = packedWeights[phase][nTupleID];
                fs.Write(BitConverter.GetBytes(pw.Length));
                foreach (var v in pw)
                {
                    if (typeof(WeightType) == typeof(Half))
                        fs.Write(BitConverter.GetBytes(Half.CreateChecked(v)));
                    else if (typeof(WeightType) == typeof(float))
                        fs.Write(BitConverter.GetBytes(float.CreateChecked(v)));
                    else if (typeof(WeightType) == typeof(double))
                        fs.Write(BitConverter.GetBytes(double.CreateChecked(v)));
                }
            }
        }

        for (var phase = 0; phase < Bias.Length; phase++)
        {
            if (typeof(WeightType) == typeof(Half))
                fs.Write(BitConverter.GetBytes(Half.CreateChecked(Bias[phase])));
            else if (typeof(WeightType) == typeof(float))
                fs.Write(BitConverter.GetBytes(float.CreateChecked(Bias[phase])));
            else if (typeof(WeightType) == typeof(double))
                fs.Write(BitConverter.GetBytes(double.CreateChecked(Bias[phase])));
        }
    }

    /// <summary>
    /// Packs weights by removing symmetric duplicates using mirror transformation.
    /// </summary>
    /// <returns>A 3D array of packed weights [phase][nTuple][feature]</returns>
    WeightType[][][] PackWeights()
    {
        var packedWeights = new List<WeightType>[NumPhases][];
        for (var i = 0; i < packedWeights.Length; i++)
            packedWeights[i] = (from _ in Enumerable.Range(0, NTupleManager.NumNTuples) select new List<WeightType>()).ToArray();

        var numPossibleFeatures = NTupleManager.NumFeatures;
        for (var phase = 0; phase < NumPhases; phase++)
        {
            for (var nTupleID = 0; nTupleID < _nTupleOffset.Length; nTupleID++)
            {
                var w = Weights.AsSpan(_phaseOffset[phase] + _nTupleOffset[nTupleID], numPossibleFeatures[nTupleID]);
                var pw = packedWeights[phase][nTupleID];
                ReadOnlySpan<int> mirror = NTupleManager.GetMirroredFeatureTable(nTupleID);
                for (var feature = 0; feature < w.Length; feature++)
                    if (feature <= mirror[feature])
                        pw.Add(w[feature]);
            }
        }

        var ret = new WeightType[NumPhases][][];
        for (var i = 0; i < ret.Length; i++)
            ret[i] = packedWeights[i].Select(n => n.ToArray()).ToArray();
        return ret;
    }

    /// <summary>
    /// Expands packed weights by restoring symmetric duplicates using mirror transformation.
    /// </summary>
    /// <param name="packedWeights">The packed weights to expand</param>
    /// <returns>A flat array of expanded weights</returns>
    WeightType[] ExpandPackedWeights(WeightType[][][] packedWeights)
    {
        var numPhases = packedWeights.Length;
        var weights = new WeightType[2 * numPhases * NTupleManager.NumFeatures.Sum()];
        for (var phase = 0; phase < numPhases; phase++)
        {
            for (var nTupleID = 0; nTupleID < _nTupleOffset.Length; nTupleID++)
            {
                var w = weights.AsSpan(_phaseOffset[phase] + _nTupleOffset[nTupleID], NTupleManager.NumFeatures[nTupleID]);
                var pw = packedWeights[phase][nTupleID];
                ReadOnlySpan<int> mirror = NTupleManager.GetMirroredFeatureTable(nTupleID);
                var i = 0;
                for (var feature = 0; feature < w.Length; feature++)
                {
                    var mirrored = mirror[feature];
                    w[feature] = (feature <= mirrored) ? pw[i++] : w[mirrored];
                }
            }
        }
        return weights;
    }
}