namespace ACG2025_reference_implementation.Evaluation;

internal static class ValueFunctionConstantConfig
{
    /// <summary>
    /// Label written to value function parameter files. Used for endianness checking.
    /// </summary>
    public const string Label = "KalmiaZero";

    /// <summary>
    /// Inversed version of the label, used to detect byte order differences.
    /// </summary>
    public const string LabelReversed = "oreZaimlaK";
    
    /// <summary>
    /// Size of the label in bytes.
    /// </summary>
    public const int LabelSize = 10;
}