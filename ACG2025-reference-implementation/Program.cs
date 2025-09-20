using System;
using ACG2025_reference_implementation.Evaluation;
using ACG2025_reference_implementation.Learn;

namespace ACG2025_reference_implementation
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var valueFunc = ValueFunctionForTrain<float>.LoadFromFile(args[0]);
            valueFunc = new ValueFunctionForTrain<float>(valueFunc.NTupleManager, 60);
            var tdTrainer = new TDTrainer<float>("TestAgent", valueFunc, new TDTrainerConfig());
            tdTrainer.Train();
        }
    }
}
