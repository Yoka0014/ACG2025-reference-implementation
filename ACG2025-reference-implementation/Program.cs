#define DEV_TEST

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using ACG2025_reference_implementation.Engines;
using ACG2025_reference_implementation.Evaluation;
using ACG2025_reference_implementation.Learn;
using ACG2025_reference_implementation.Protocols;

namespace ACG2025_reference_implementation
{
    internal class Program
    {
        static void Main(string[] args)
        {
#if MCTS_ENGINE
            EngineMain(new MCTSEngine());
#elif ALPHA_BETA_ENGINE
            EngineMain(new AlphaBetaPruningEngine());
#elif OPTIMIZE_NTUPLE
            OptimizeNTupleMain(args);
#elif RL_SELFPLAY
            RLSelfPlayMain(args);
#elif DECODE_POOL
            DecodePoolMain(args);
#elif DEV_TEST
            DevTest(args);
#endif
        }

        static void EngineMain(Engine engine)
        {
            var nboard = new NBoard();
            nboard.Mainloop(engine);            
        }

        static void OptimizeNTupleMain(string[] args)
        {
            const int NumArgs = 7;

            if (args.Length < NumArgs - 1)
            {
                Console.WriteLine("**Hints**");
                Console.WriteLine("args[0]: path to main configuration JSON file.");
                Console.WriteLine("args[1]: path to BRKGA configuration file.");
                Console.WriteLine("args[2]: path to TD training configuration file.");
                Console.WriteLine("args[3]: the size of n-tuple.");
                Console.WriteLine("args[4]: the number of n-tuples.");
                Console.WriteLine("args[5]: the number of generations.");
                Console.WriteLine("args[6]: (optional) path to initial pool file.");
                Console.WriteLine();
                Console.WriteLine("Examples:");
                Console.WriteLine("  From scratch: ./*.exe ntuple_optimizer_config.json brkga_config.json td_config.json 10 12");
                Console.WriteLine("  Continuing with existing pool: ./*.exe ntuple_optimizer_config.json brkga_config.json td_config.json 10 12 pool.bin");
                return;
            }

            var mainConfig = LoadConfig<NTupleOptimizerConfig>(args[0]);

            if (mainConfig is null)
                return;

            var gaConfig = LoadConfig<BRKGAConfig>(args[1]);

            if (gaConfig is null)
                return;

            var tdConfig = LoadConfig<TDTrainerConfig>(args[2]);

            if (tdConfig is null)
                return;

            var nTupleSize = ParsePositiveInt(args[3], "The size of n-tuple.");

            if (!nTupleSize.HasValue)
                return;

            var numNTuples = ParsePositiveInt(args[4], "The number of n-tuples.");

            if (!numNTuples.HasValue)
                return;

            var numGenerations = ParsePositiveInt(args[5], "The number of generations");

            if (!numGenerations.HasValue)
                return;

            var optimizer = new NTupleOptimizer<float>(mainConfig, gaConfig, tdConfig, nTupleSize.Value, numNTuples.Value);

            if (args.Length < NumArgs)
                optimizer.Train(numGenerations.Value);
            else
                optimizer.Train(args[^1], numGenerations.Value);
        }

        static void RLSelfPlayMain(string[] args)
        {
            const int NumArgs = 4;
            if (args.Length < NumArgs - 1)
            {
                Console.WriteLine("**Hints**");
                Console.WriteLine("args[0]: path to configuration JSON file.");
                Console.WriteLine("args[1]: path to value function's weights file.");
                Console.WriteLine("args[2]: the number of training cycles.");
                Console.WriteLine("args[3]: (optional) \"zero\" to initialize weights with zeros, or omit to use existing weights");
                Console.WriteLine();
                Console.WriteLine("Examples:");
                Console.WriteLine("  Use existing weights: ./*.exe selfplay_config.json value_func_weights.bin 10");
                Console.WriteLine("  Initialize with zeros: ./*.exe selfplay_config.json value_func_weights.bin 10 zero");
                return;
            }

            var config = LoadConfig<SelfPlayTrainerConfig>(args[0]);

            if (config is null)
                return;

            ValueFunctionForTrain<float> valueFunc;

            if (!File.Exists(args[1]))
            {
                Console.Error.WriteLine($"Error: value function's weights file \"{args[1]}\" was not found.");
                return;
            }

            try
            {
                valueFunc = ValueFunctionForTrain<float>.LoadFromFile(args[1]);
            }
            catch (InvalidDataException ex)
            {
                Console.Error.WriteLine($"Error: Specified value function's weights file \"{args[1]}\" is not in the correct format.\n\nDetail: {ex.Message}");
                return;
            }

            var numCycles = ParsePositiveInt(args[2], "The number of training cycles.");

            if (!numCycles.HasValue)
                return;

            if (args.Length >= NumArgs)
            {
                if (args[3] == "zero")
                {
                    valueFunc.InitWithZeros();
                }
                else if (!string.IsNullOrEmpty(args[3]))
                {
                    Console.Error.WriteLine($"Error: \"{args[3]}\" is an invalid option.");
                    return;
                }
            }

            new SelfPlayTrainer<float>(config).Train(valueFunc, numCycles.Value);
        }

        static void DecodePoolMain(string[] args)
        {
            const int NumArgs = 5;
            if (args.Length < NumArgs - 1)
            {
                Console.WriteLine("**Hints**");
                Console.WriteLine("args[0]: path to pool file.");
                Console.WriteLine("args[1]: the number of individual to decode from the top.");
                Console.WriteLine("args[2]: the size of n-tuple.");
                Console.WriteLine("args[3]: the number of n-tuples.");
                Console.WriteLine("args[4]: (Optional) the number of moves in a phase of value function (default: 60).");
                Console.WriteLine("Examples:");
                Console.WriteLine("  When decoding top-3 individuals and getting 12 10-tuples: ./*.exe pool.bin 3 10 12");
                return;
            }

            if (!File.Exists(args[0]))
            {
                Console.Error.WriteLine($"Error: pool file \"{args[0]}\" was not found.");
                return;
            }

            Individual[] pool;
            try
            {
                pool = Individual.LoadPoolFromFile(args[0]);
            }
            catch (InvalidDataException)
            {
                Console.Error.WriteLine($"Error: pool file \"{args[0]}\" is not in the correct format.");
                return;
            }

            var count = ParsePositiveInt(args[1], "The number of individuals to be decoded");

            if (!count.HasValue)
                return;

            var nTupleSize = ParsePositiveInt(args[2], "The size of n-tuple");

            if (!nTupleSize.HasValue)
                return;

            var numNTuples = ParsePositiveInt(args[3], "The number of n-tuples");

            if (!numNTuples.HasValue)
                return;

            int? numMovesPerPhase = 60;
            if (args.Length >= NumArgs)
            {
                numMovesPerPhase = ParsePositiveInt(args[4], "The number of moves in the phase");

                if (numMovesPerPhase is null)
                    return;
            }

            Array.Sort(pool, Comparer<Individual>.Create((x, y) => y.Fitness.CompareTo(x.Fitness)));

            count = Math.Min(count.Value, pool.Length);
            foreach (var (i, nTupleManager) in BRKGA<float>.DecodePool(pool, nTupleSize.Value, numNTuples.Value, count.Value).Select((v, i) => (i, v)))
            {
                var valueFunc = new ValueFunctionForTrain<float>(nTupleManager, numMovesPerPhase.Value);
                valueFunc.SaveToFile($"value_func_weights_idv{i}.bin");

                using var sw = new StreamWriter($"ntuples_idv{i}.txt");
                foreach (var nTuple in nTupleManager.NTuples)
                {
                    sw.Write('[');
                    var coords = nTuple.GetCoordinates(0);
                    foreach (var coord in coords[..^1])
                        sw.Write($"{(int)coord}, ");
                    sw.Write($"{coords[^1]}]\n");
                }
            }

            Console.WriteLine($"{count} individuals were decoded.");
        }

        static T? LoadConfig<T>(string path)
        {
            if (!File.Exists(path))
            {
                Console.Error.WriteLine($"Error: Configuration file \"{path}\" was not found.");
                return default;
            }

            T? config;
            try
            {
                config = JsonSerializer.Deserialize<T>(File.ReadAllText(path));
            }
            catch (JsonException ex)
            {
                Console.Error.WriteLine($"Error: Configuration file \"{path}\" is not in the correct JSON format.\n\nDetail: {ex.Message}");
                return default;
            }

            if (config is null)
            {
                Console.Error.WriteLine($"Error: Configuration cannot be null. This may be caused by an empty JSON file.");
                return default;
            }

            return config;
        }

        static int? ParsePositiveInt(string str, string label)
        {
            if (!int.TryParse(str, out var value) || value <= 0)
            {
                Console.Error.WriteLine($"Error: {label} must be a positive integer less than or equal to {int.MaxValue}.");
                return null;
            }
            return value;
        }

        static void DevTest(string[] args)
        {
        }
    }
}