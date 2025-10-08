using System;
using System.IO;
using System.Text.Json;
using ACG2025_reference_implementation.Engines;
using ACG2025_reference_implementation.Evaluation;
using ACG2025_reference_implementation.Learn;
using ACG2025_reference_implementation.Protocols;
using ACG2025_reference_implementation.Reversi;

namespace ACG2025_reference_implementation
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var option = new JsonSerializerOptions { WriteIndented = true };
            File.WriteAllText("brkga_config.json", JsonSerializer.Serialize(new BRKGAConfig(), option));
            File.WriteAllText("td_config.json", JsonSerializer.Serialize(new TDTrainerConfig(), option));
            File.WriteAllText("selfplay_config.json", JsonSerializer.Serialize(new SelfPlayTrainerConfig(), option));
            File.WriteAllText("ntuple_optimizer_config.json", JsonSerializer.Serialize(new NTupleOptimizerConfig(), option));
        }
    }
}
