namespace ACG2025_reference_implementation.Learn;

using System;
using System.IO;
using System.Text;
using System.Numerics;
using System.Linq;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

using ACG2025_reference_implementation.Utils;
using ACG2025_reference_implementation.Reversi;
using ACG2025_reference_implementation.NTupleSystem;
using ACG2025_reference_implementation.Evaluation;

/// <summary>
/// Configuration parameters for the Biased Random-Key Genetic Algorithm (BRKGA).
/// Contains genetic algorithm parameters such as population size, elite and mutant rates,
/// and evaluation parameters for fitness calculation.
/// </summary>
internal record class BRKGAConfig
{
    /// <summary>
    /// Gets or sets the number of individuals in the population.
    /// </summary>
    /// <value>Default is 100 individuals</value>
    public int PopulationSize { get; init; } = 100;
    
    /// <summary>
    /// Gets or sets the proportion of elite individuals in the population.
    /// Elite individuals are the best-performing individuals from the previous generation.
    /// </summary>
    /// <value>Default is 0.2 (20% of population)</value>
    public double EliteRate { get; init; } = 0.2;
    
    /// <summary>
    /// Gets or sets the proportion of mutant individuals in the population.
    /// Mutants are randomly generated individuals that provide genetic diversity.
    /// </summary>
    /// <value>Default is 0.2 (20% of population)</value>
    public double MutantRate { get; init; } = 0.2;
    
    /// <summary>
    /// Gets or sets the probability of inheriting genes from the elite parent during crossover.
    /// Higher values favor elite characteristics in offspring.
    /// </summary>
    /// <value>Default is 0.7 (70% probability)</value>
    public double EliteInheritanceProb { get; init; } = 0.7;

    /// <summary>
    /// Gets or sets the learning rate used for supervised learning during fitness evaluation.
    /// </summary>
    /// <value>Default is 0.2</value>
    public double LearningRateForEval { get; init; } = 0.2;
    
    /// <summary>
    /// Gets or sets the number of training epochs used during fitness evaluation.
    /// </summary>
    /// <value>Default is 20 epochs</value>
    public int NumEpochsForEval { get; init; } = 20;

    /// <summary>
    /// Gets or sets the base filename for saving population pools.
    /// Generation number will be appended to create unique filenames.
    /// </summary>
    /// <value>Default is "pool"</value>
    public string PoolFileName { get; init; } = "pool";
    
    /// <summary>
    /// Gets or sets the filename for saving fitness history data.
    /// </summary>
    /// <value>Default is "fitness_history"</value>
    public string FitnessHistoryFileName { get; init; } = "fitness_history";
}

/// <summary>
/// Represents an individual in the BRKGA population.
/// Each individual contains a chromosome (array of random keys) and a fitness value.
/// Implements IComparable for sorting individuals by fitness in descending order.
/// </summary>
internal struct Individual : IComparable<Individual>
{
    /// <summary>
    /// File format identifier used for endianness detection during serialization.
    /// </summary>
    const string Label = "KalmiaZero_Pool";
    
    /// <summary>
    /// Reversed label used for detecting byte order in serialized files.
    /// </summary>
    const string LabelReversed = "looP_oreZaimlaK";
    
    /// <summary>
    /// Size of the label in bytes.
    /// </summary>
    const int LabelSize = 15;

    /// <summary>
    /// Gets the chromosome array containing random keys that encode the n-tuple configuration.
    /// Each gene is a floating-point value between 0 and 1.
    /// </summary>
    public float[] Chromosome { get; private set; }
    
    /// <summary>
    /// Gets or sets the fitness value of this individual.
    /// Higher fitness indicates better performance.
    /// </summary>
    public float Fitness { get; set; }

    /// <summary>
    /// Initializes a new individual with a randomly generated chromosome of the specified size.
    /// Uses the shared random number generator.
    /// </summary>
    /// <param name="chromSize">The size of the chromosome (number of genes)</param>
    public Individual(int chromSize) : this(chromSize, Random.Shared) { }

    /// <summary>
    /// Initializes a new individual with a randomly generated chromosome using the specified random generator.
    /// </summary>
    /// <param name="chromSize">The size of the chromosome (number of genes)</param>
    /// <param name="rand">The random number generator to use for chromosome initialization</param>
    public Individual(int chromSize, Random rand)
    {
        Chromosome = [.. Enumerable.Range(0, chromSize).Select(_ => rand.NextSingle())];
        Fitness = float.NegativeInfinity;
    }

    /// <summary>
    /// Copy constructor that creates a deep copy of the specified individual.
    /// </summary>
    /// <param name="src">The source individual to copy from</param>
    public Individual(ref Individual src)
    {
        Fitness = src.Fitness;
        Chromosome = new float[src.Chromosome.Length];
        Buffer.BlockCopy(src.Chromosome, 0, Chromosome, 0, sizeof(float) * Chromosome.Length);
    }

    /// <summary>
    /// Initializes an individual by deserializing data from a stream.
    /// </summary>
    /// <param name="stream">The stream to read individual data from</param>
    /// <param name="swapBytes">Whether to swap byte order during deserialization</param>
    public Individual(Stream stream, bool swapBytes)
    {
        const int BufferSize = 4;

        Span<byte> buffer = stackalloc byte[BufferSize];
        stream.Read(buffer[..sizeof(int)], swapBytes);
        Chromosome = new float[BitConverter.ToInt32(buffer)];
        for (var i = 0; i < Chromosome.Length; i++)
        {
            stream.Read(buffer[..sizeof(float)], swapBytes);
            Chromosome[i] = BitConverter.ToSingle(buffer);
        }

        stream.Read(buffer[..sizeof(float)], swapBytes);
        Fitness = BitConverter.ToSingle(buffer);
    }

    public readonly void CopyTo(ref Individual dest)
    {
        dest.Fitness = Fitness;
        Buffer.BlockCopy(Chromosome, 0, dest.Chromosome, 0, sizeof(float) * Chromosome.Length);
    }

    /// <summary>
    /// Serializes this individual to the specified stream.
    /// Writes chromosome length, chromosome data, and fitness value in binary format.
    /// </summary>
    /// <param name="stream">The stream to write the individual data to</param>
    public readonly void WriteTo(Stream stream)
    {
        stream.Write(BitConverter.GetBytes(Chromosome.Length));
        foreach (var gene in Chromosome)
            stream.Write(BitConverter.GetBytes(gene));
        stream.Write(BitConverter.GetBytes(Fitness));
    }

    /// <summary>
    /// Compares this individual with another individual based on fitness values.
    /// Implements descending order comparison (higher fitness comes first).
    /// </summary>
    /// <param name="other">The individual to compare with</param>
    /// <returns>Negative value if this individual has lower fitness, positive if higher, zero if equal</returns>
    public readonly int CompareTo(Individual other) => other.Fitness.CompareTo(Fitness);

    /// <summary>
    /// Loads a population pool from a binary file.
    /// </summary>
    /// <param name="path">The path to the pool file</param>
    /// <returns>An array of individuals loaded from the file</returns>
    /// <exception cref="InvalidDataException">Thrown when the file format is invalid</exception>
    /// <remarks>
    /// File format:
    /// - offset 0: LABEL (15 bytes)
    /// - offset 15: POPULATION_SIZE (4 bytes)
    /// - offset 27: INDIVIDUAL[0] data
    /// - ...
    /// </remarks>
    public static Individual[] LoadPoolFromFile(string path)
    {
        Span<byte> buffer = stackalloc byte[LabelSize];
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
        fs.Read(buffer);
        var label = Encoding.ASCII.GetString(buffer);
        var swapBytes = label == LabelReversed;

        if (!swapBytes && label != Label)
            throw new InvalidDataException($"The format of \"{path}\" is invalid.");

        fs.Read(buffer[..sizeof(int)], swapBytes);
        var populationSize = BitConverter.ToInt32(buffer);
        return Enumerable.Range(0, populationSize).Select(_ => new Individual(fs, swapBytes)).ToArray();
    }

    /// <summary>
    /// Saves a population pool to a binary file at the specified path.
    /// </summary>
    /// <param name="pool">The array of individuals to save</param>
    /// <param name="path">The file path where the pool will be saved</param>
    public static void SavePoolAt(Individual[] pool, string path)
    {
        Span<byte> buffer = stackalloc byte[LabelSize];
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
        Encoding.ASCII.GetBytes(Label).CopyTo(buffer);
        fs.Write(buffer);
        fs.Write(BitConverter.GetBytes(pool.Length));
        for (var i = 0; i < pool.Length; i++)
            pool[i].WriteTo(fs);
    }
}

/// <summary>
/// Implements the Biased Random-Key Genetic Algorithm (BRKGA) for evolving n-tuples configurations
/// for Reversi position evaluation. The algorithm uses random keys to encode n-tuple coordinates and
/// evolves populations to find optimal feature combinations.
/// </summary>
/// <typeparam name="WeightType">The floating-point type used for value function's weights (Half, float, or double)</typeparam>
internal class BRKGA<WeightType> where WeightType : unmanaged, IFloatingPointIeee754<WeightType>
{
    /// <summary>BRKGA configuration parameters</summary>
    readonly BRKGAConfig _config;
    
    /// <summary>Supervised learning configuration for fitness evaluation</summary>
    readonly SupervisedTrainerConfig _slConfig;
    
    /// <summary>File path template for saving population pools</summary>
    readonly string _poolPath;
    
    /// <summary>File path for saving fitness history</summary>
    readonly string _fitnessHistoryPath;
    
    /// <summary>Number of elite individuals per generation</summary>
    readonly int _numElites;
    
    /// <summary>Number of mutant individuals per generation</summary>
    readonly int _numMutants;
    
    /// <summary>Random number generator for genetic operations</summary>
    readonly Random _rand;

    /// <summary>Parallel processing configuration</summary>
    ParallelOptions _parallelOptions = new() { MaxDegreeOfParallelism = Environment.ProcessorCount };

    /// <summary>Size of each n-tuple (number of board coordinates)</summary>
    readonly int _nTupleSize;
    
    /// <summary>Number of n-tuples per individual</summary>
    readonly int _numNTuples;

    /// <summary>Current population of individuals</summary>
    Individual[] _pool;
    
    /// <summary>Next generation population buffer</summary>
    Individual[] _nextPool;

    /// <summary>Logger for training output</summary>
    StreamWriter? _logger;

    /// <summary>History of fitness statistics across generations</summary>
    readonly List<(float best, float worst, float median, float average)> _fitnessHistory = [];

    /// <summary>
    /// Initializes a new BRKGA instance with the specified configuration and n-tuple parameters.
    /// Uses a randomly seeded random number generator.
    /// </summary>
    /// <param name="config">BRKGA configuration parameters</param>
    /// <param name="nTupleSize">Size of each n-tuple (number of coordinates)</param>
    /// <param name="numNTuples">Number of n-tuples per individual</param>
    public BRKGA(BRKGAConfig config, int nTupleSize, int numNTuples) : this(config, nTupleSize, numNTuples, new Random(Random.Shared.Next())) { }

    /// <summary>
    /// Initializes a new BRKGA instance with the specified configuration, n-tuple parameters, and random generator.
    /// </summary>
    /// <param name="config">BRKGA configuration parameters</param>
    /// <param name="nTupleSize">Size of each n-tuple (number of coordinates)</param>
    /// <param name="numNTuples">Number of n-tuples per individual</param>
    /// <param name="rand">Random number generator for genetic operations</param>
    public BRKGA(BRKGAConfig config, int nTupleSize, int numNTuples, Random rand)
    {
        _config = config;

        _slConfig = new SupervisedTrainerConfig
        {
            NumEpoch = config.NumEpochsForEval,
            LearningRate = config.LearningRateForEval,
            WeightsFileName = string.Empty,
            LossHistoryFileName = string.Empty,
            SaveWeightsInterval = int.MaxValue
        };

        _poolPath = $"{config.PoolFileName}_{"{0}"}.bin";
        _fitnessHistoryPath = $"{config.FitnessHistoryFileName}.log";

        _numElites = (int)(config.PopulationSize * config.EliteRate);
        _numMutants = (int)(config.PopulationSize * config.MutantRate);
        _nTupleSize = nTupleSize;
        _numNTuples = numNTuples;
        _rand = rand;

        _pool = [.. Enumerable.Range(0, _config.PopulationSize).Select(_ => new Individual(Constants.NumCells * _numNTuples))];
        _nextPool = [.. Enumerable.Range(0, _config.PopulationSize).Select(_ => new Individual(Constants.NumCells * _numNTuples))];
    }

    /// <summary>
    /// Gets or sets the maximum number of threads used for parallel processing during fitness evaluation.
    /// </summary>
    public int NumThreads
    {
        get => _parallelOptions.MaxDegreeOfParallelism;
        set => _parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = value };
    }

    /// <summary>
    /// Gets a copy of the current population pool.
    /// </summary>
    /// <returns>An array containing copies of all individuals in the current population</returns>
    public Individual[] GetCurrentPool() => [.. _pool.Select(i => new Individual(ref i))];

    /// <summary>
    /// Decodes the current population into n-tuple managers.
    /// Converts each individual's chromosome into a concrete n-tuple configuration.
    /// </summary>
    /// <returns>An array of n-tuple managers corresponding to each individual in the population</returns>
    public NTupleManager[] DecodeCurrentPool()
    {
        var nTupleManagers = new NTupleManager[_pool.Length];
        for (var i = 0; i < _pool.Length; i++)
            nTupleManagers[i] = new NTupleManager(DecodeChromosome(_pool[i].Chromosome, _nTupleSize, _numNTuples));
        return nTupleManagers;
    }

    /// <summary>
    /// Trains the BRKGA with a randomly initialized population.
    /// </summary>
    /// <param name="trainData">Training dataset for fitness evaluation</param>
    /// <param name="testData">Test dataset for fitness evaluation</param>
    /// <param name="numGenerations">Number of generations to evolve</param>
    public void Train(GameDataset trainData, GameDataset testData, int numGenerations)
        => Train([.. Enumerable.Range(0, _config.PopulationSize).Select(_ => new Individual(Constants.NumCells * _numNTuples, _rand))], trainData, testData, numGenerations, Console.OpenStandardOutput());

    /// <summary>
    /// Trains the BRKGA starting from a population loaded from a file.
    /// </summary>
    /// <param name="poolPath">Path to the pool file containing the initial population</param>
    /// <param name="trainData">Training dataset for fitness evaluation</param>
    /// <param name="testData">Test dataset for fitness evaluation</param>
    /// <param name="numGenerations">Number of generations to evolve</param>
    public void Train(string poolPath, GameDataset trainData, GameDataset testData, int numGenerations)
        => Train(Individual.LoadPoolFromFile(poolPath), trainData, testData, numGenerations, Console.OpenStandardOutput());

    /// <summary>
    /// Trains the BRKGA with the specified initial population and logging stream.
    /// </summary>
    /// <param name="initialPool">Initial population of individuals</param>
    /// <param name="trainData">Training dataset for fitness evaluation</param>
    /// <param name="testData">Test dataset for fitness evaluation</param>
    /// <param name="numGenerations">Number of generations to evolve</param>
    public void Train(Individual[] initialPool, GameDataset trainData, GameDataset testData, int numGenerations)
        => Train(initialPool, trainData, testData, numGenerations, Console.OpenStandardOutput());

    /// <summary>
    /// Trains the BRKGA with the specified initial population and logging stream.
    /// </summary>
    /// <param name="initialPool">Initial population of individuals</param>
    /// <param name="trainData">Training dataset for fitness evaluation</param>
    /// <param name="testData">Test dataset for fitness evaluation</param>
    /// <param name="numGenerations">Number of generations to evolve</param>
    /// <param name="logStream">Stream for logging training progress</param>
    public void Train(Individual[] initialPool, GameDataset trainData, GameDataset testData, int numGenerations, Stream logStream)
    {
        _logger = new StreamWriter(logStream) { AutoFlush = true };

        CopyPool(initialPool, _pool);
        CopyPool(initialPool, _nextPool);

        for (var i = 0; i < _pool.Length; i++)
            _pool[i].Fitness = float.NegativeInfinity;

        TrainWithCurrentPool(trainData, testData, numGenerations);
    }

    /// <summary>
    /// Continues training with the current population pool for the specified number of generations.
    /// Performs evolutionary operations including fitness evaluation, selection, crossover, and mutation.
    /// </summary>
    /// <param name="trainData">Training dataset for fitness evaluation</param>
    /// <param name="testData">Test dataset for fitness evaluation</param>
    /// <param name="numGenerations">Number of additional generations to evolve</param>
    public void TrainWithCurrentPool(GameDataset trainData, GameDataset testData, int numGenerations)
    {
        for (var gen = 0; gen < numGenerations; gen++)
        {
            _logger?.WriteLine($"Generation: {gen}");

            EvaluatePool(trainData, testData);

            var nonMutants = _pool[..^_numMutants];
            _fitnessHistory.Add((_pool[0].Fitness, nonMutants[^1].Fitness,
                                 nonMutants[nonMutants.Length / 2].Fitness, nonMutants.Average(p => p.Fitness)));

            var (best, worst, median, average) = _fitnessHistory[^1];
            _logger?.WriteLine($"\nBestFitness: {best}");
            _logger?.WriteLine($"WorstFitness: {worst}");
            _logger?.WriteLine($"MedianFitness: {median}");
            _logger?.WriteLine($"AverageFitness: {average}");

            Individual.SavePoolAt(_pool, string.Format(_poolPath, gen));
            SaveFitnessHistory();

            _logger?.WriteLine("Generate individuals for next generation.");

            var elites = _pool.AsSpan(0, _numElites);
            var nonElites = _pool.AsSpan(_numElites);
            CopyPool(elites, _nextPool);

            GenerateChildren(ref elites, ref nonElites);
            GenerateMutants();

            (_pool, _nextPool) = (_nextPool, _pool);

            _logger?.WriteLine();
        }

        _logger?.WriteLine("Final evaluation.");
        EvaluatePool(trainData, testData);
    }

    /// <summary>
    /// Saves the fitness history to a file in Python list format for easy visualization.
    /// Records best, worst, median, and average fitness values across all generations.
    /// </summary>
    void SaveFitnessHistory()
    {
        // Save in Python list format for easy visualization with Python + matplotlib.
        var bestSb = new StringBuilder("[");
        var worstSb = new StringBuilder("[");
        var medianSb = new StringBuilder("[");
        var averageSb = new StringBuilder("[");
        foreach ((var best, var worst, var median, var average) in _fitnessHistory)
        {
            bestSb.Append(best).Append(", ");
            worstSb.Append(worst).Append(", ");
            medianSb.Append(median).Append(", ");
            averageSb.Append(average).Append(", ");
        }

        // remove last ", ";
        bestSb.Remove(bestSb.Length - 2, 2);
        worstSb.Remove(worstSb.Length - 2, 2);
        medianSb.Remove(medianSb.Length - 2, 2);
        averageSb.Remove(averageSb.Length - 2, 2);

        bestSb.Append(']');
        worstSb.Append(']');
        medianSb.Append(']');
        averageSb.Append(']');

        using var sw = new StreamWriter(_fitnessHistoryPath);
        sw.WriteLine(bestSb.ToString());
        sw.WriteLine(worstSb.ToString());
        sw.WriteLine(medianSb.ToString());
        sw.WriteLine(averageSb.ToString());
    }

    /// <summary>
    /// Evaluates the fitness of all individuals in the population in parallel.
    /// Only evaluates individuals with uninitialized fitness values and sorts the pool by fitness afterwards.
    /// </summary>
    /// <param name="trainData">Training dataset used for supervised learning</param>
    /// <param name="testData">Test dataset used for final fitness calculation</param>
    void EvaluatePool(GameDataset trainData, GameDataset testData)
    {
        _logger?.WriteLine("Start evaluation.");

        var count = 0;
        Parallel.For(0, _pool.Length, _parallelOptions, i =>
        {
            if (float.IsNegativeInfinity(_pool[i].Fitness))
                EvaluateIndividual(ref _pool[i], trainData, testData, i);
            Interlocked.Increment(ref count);
            _logger?.WriteLine($"{count} individuals were evaluated({count * 100.0 / _config.PopulationSize:f2}%).");
        });
        Array.Sort(_pool);
    }

    /// <summary>
    /// Generates mutant individuals by randomly initializing their chromosomes.
    /// Mutants provide genetic diversity and help prevent premature convergence.
    /// </summary>
    void GenerateMutants()
    {
        var mutants = _nextPool.AsSpan(_nextPool.Length - _numMutants);
        for (var i = 0; i < mutants.Length; i++)
        {
            ref var mutant = ref mutants[i];
            for (var j = 0; j < mutant.Chromosome.Length; j++)
                mutant.Chromosome[j] = _rand.NextSingle();
            mutant.Fitness = float.NegativeInfinity;
        }
    }

    /// <summary>
    /// Generates offspring through crossover between elite and non-elite parents.
    /// Each child is produced by crossing one randomly selected elite with one randomly selected non-elite.
    /// </summary>
    /// <param name="elites">Span of elite individuals (best performers)</param>
    /// <param name="nonElites">Span of non-elite individuals</param>
    void GenerateChildren(ref Span<Individual> elites, ref Span<Individual> nonElites)
    {
        var numChildren = _config.PopulationSize - _numElites - _numMutants;
        var children = _nextPool.AsSpan(_numElites, numChildren);
        for (var i = 0; i < children.Length; i++)
            Crossover(ref elites[_rand.Next(elites.Length)], ref nonElites[_rand.Next(nonElites.Length)], ref children[i]);
    }

    /// <summary>
    /// Performs biased crossover between an elite parent and a non-elite parent.
    /// Each gene in the child chromosome is inherited from the elite parent with probability EliteInheritanceProb,
    /// otherwise it comes from the non-elite parent.
    /// </summary>
    /// <param name="eliteParent">The elite parent individual</param>
    /// <param name="nonEliteParent">The non-elite parent individual</param>
    /// <param name="child">The child individual to be created</param>
    void Crossover(ref Individual eliteParent, ref Individual nonEliteParent, ref Individual child)
    {
        (var eliteChrom, var nonEliteChrom) = (eliteParent.Chromosome, nonEliteParent.Chromosome);
        var childChrom = child.Chromosome;
        for (var i = 0; i < childChrom.Length; i++)
            childChrom[i] = (_rand.NextDouble() < _config.EliteInheritanceProb) ? eliteChrom[i] : nonEliteChrom[i];
        child.Fitness = float.NegativeInfinity;
    }

    /// <summary>
    /// Evaluates the fitness of an individual by training its n-tuple network and measuring performance.
    /// Decodes the chromosome into n-tuples, trains the resulting value function, and calculates fitness
    /// as the reciprocal of the test loss (higher fitness = lower loss).
    /// </summary>
    /// <param name="individual">The individual to evaluate</param>
    /// <param name="trainData">Training dataset for supervised learning</param>
    /// <param name="testData">Test dataset for fitness calculation</param>
    /// <param name="id">Unique identifier for the individual (used in logging)</param>
    void EvaluateIndividual(ref Individual individual, GameDataset trainData, GameDataset testData, int id)
    {
        var nTupleManager = new NTupleManager(DecodeChromosome(individual.Chromosome, _nTupleSize, _numNTuples));
        var valueFunc = new ValueFunctionForTrain<WeightType>(nTupleManager);
        var slTrainer = new SupervisedTrainer<WeightType>($"INDV_{id}", valueFunc, _slConfig, Stream.Null);
        
        slTrainer.Train(trainData, [], 1, saveWeights: false, saveLossHistory: false);
        individual.Fitness = float.CreateChecked(WeightType.One / CalculateLoss(valueFunc, testData));
    }

    static void CopyPool(Span<Individual> src, Span<Individual> dest)
    {
        for (var i = 0; i < src.Length; i++)
            src[i].CopyTo(ref dest[i]);
    }

    /// <summary>
    /// Calculates the binary cross-entropy loss of a value function on the given dataset.
    /// Iterates through all positions in all games and computes the average loss.
    /// </summary>
    /// <param name="valueFunc">The value function to evaluate</param>
    /// <param name="trainData">The dataset to calculate loss on</param>
    /// <returns>The average binary cross-entropy loss across all positions</returns>
    static WeightType CalculateLoss(ValueFunctionForTrain<WeightType> valueFunc, GameDataset trainData)
    {
        var loss = WeightType.Zero;
        var count = 0;
        var featureVec = new FeatureVector(valueFunc.NTupleManager);
        for (var i = 0; i < trainData.Length; i++)
        {
            ref var data = ref trainData[i];
            var pos = data.RootPos;

            featureVec.Init(ref pos);

            for (var j = 0; j < data.Moves.Length; j++)
            {
                var reward = GetReward(ref data, pos.SideToMove, pos.EmptyCellCount);
                loss += MathFunctions.BinaryCrossEntropy(valueFunc.PredictWithBlackWeights(featureVec), reward);
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
        return loss / WeightType.CreateChecked(count);
    }

    /// <summary>
    /// Converts the game outcome to a reward signal for the current side to move.
    /// Returns 1.0 for a win, 0.0 for a loss, and 0.5 for a draw.
    /// </summary>
    /// <param name="data">The game data item containing the final score</param>
    /// <param name="sideToMove">The side to move at the current position</param>
    /// <param name="emptySquareCount">Number of empty squares (unused but kept for interface compatibility)</param>
    /// <returns>Reward value: 1.0 (win), 0.5 (draw), or 0.0 (loss)</returns>
    static WeightType GetReward(ref GameDatasetItem data, DiscColor sideToMove, int emptySquareCount)
    {
        var score = data.ScoreFromBlack;

        if (sideToMove != DiscColor.Black)
            score *= -1;

        if (score == 0)
            return WeightType.One / (WeightType.One + WeightType.One);

        return (score > 0) ? WeightType.One : WeightType.Zero;
    }

    /// <summary>
    /// Decodes a population pool from a file into n-tuple managers.
    /// </summary>
    /// <param name="poolPath">Path to the pool file</param>
    /// <param name="nTupleSize">Size of each n-tuple</param>
    /// <param name="numNTuples">Number of n-tuples per individual</param>
    /// <param name="numIndividuals">Number of individuals to decode (-1 for all)</param>
    /// <returns>Array of n-tuple managers decoded from the pool</returns>
    public static NTupleManager[] DecodePool(string poolPath, int nTupleSize, int numNTuples, int numIndividuals = -1)
    {
        var pool = Individual.LoadPoolFromFile(poolPath);
        return DecodePool(pool, nTupleSize, numNTuples, numIndividuals);
    }

    /// <summary>
    /// Decodes a population pool into n-tuple managers.
    /// Sorts the pool by fitness and decodes the best individuals.
    /// </summary>
    /// <param name="pool">Array of individuals to decode</param>
    /// <param name="nTupleSize">Size of each n-tuple</param>
    /// <param name="numNTuples">Number of n-tuples per individual</param>
    /// <param name="numIndividuals">Number of individuals to decode (-1 for all)</param>
    /// <returns>Array of n-tuple managers decoded from the pool</returns>
    public static NTupleManager[] DecodePool(Individual[] pool, int nTupleSize, int numNTuples, int numIndividuals = -1)
    {
        Array.Sort(pool);

        if (numIndividuals == -1)
            numIndividuals = pool.Length;

        var nTuplesGroup = new NTupleManager[numIndividuals];
        for (var i = 0; i < nTuplesGroup.Length; i++)
            nTuplesGroup[i] = new NTupleManager(DecodeChromosome(pool[i].Chromosome, nTupleSize, numNTuples));

        return nTuplesGroup;
    }

    /// <summary>
    /// Decodes a chromosome (array of random keys) into an array of n-tuples.
    /// Uses a greedy algorithm to select coordinates based on the minimum values in the chromosome,
    /// ensuring that selected coordinates form connected components.
    /// </summary>
    /// <param name="chromosome">The chromosome array containing random keys</param>
    /// <param name="nTupleSize">Size of each n-tuple (number of coordinates)</param>
    /// <param name="numNTuples">Number of n-tuples to generate</param>
    /// <returns>Array of decoded n-tuples</returns>
    /// <exception cref="ArgumentException">Thrown when the chromosome cannot generate the required n-tuple configuration</exception>
    static NTuple[] DecodeChromosome(float[] chromosome, int nTupleSize, int numNTuples)
    {
        var nTuples = new NTuple[numNTuples];
        var coords = new BoardCoordinate[nTupleSize];
        var adjCoords = new List<BoardCoordinate>();
        for (var nTupleID = 0; nTupleID < numNTuples; nTupleID++)
        {
            var chrom = chromosome.AsSpan(nTupleID * Constants.NumCells, Constants.NumCells);
            var min = chrom[0];
            var minIdx = 0;
            for (var i = 1; i < chrom.Length; i++)
            {
                if (chrom[i] < min)
                {
                    minIdx = i;
                    min = chrom[i];
                }
            }

            Array.Fill(coords, BoardCoordinate.Null);
            coords[0] = (BoardCoordinate)minIdx;
            adjCoords.Clear();
            for (var i = 1; i < nTupleSize; i++)
            {
                adjCoords.AddRange(Utils.GetAdjacent8Cells(coords[i - 1]));
                adjCoords.RemoveAll(coords[..i].Contains);

                min = float.PositiveInfinity;
                foreach (var adjCoord in adjCoords)
                {
                    if (chrom[(int)adjCoord] < min)
                    {
                        coords[i] = adjCoord;
                        min = chrom[(int)adjCoord];
                    }
                }

                if (float.IsPositiveInfinity(min))
                    throw new ArgumentException($"Cannot create a {nTupleSize}-Tuple from the specified chromosome.");
            }

            nTuples[nTupleID] = new NTuple(coords);
        }

        return nTuples;
    }
}