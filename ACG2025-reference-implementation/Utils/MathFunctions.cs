using System.Numerics;

namespace ACG2025_reference_implementation.Utils;

/// <summary>
/// Provides static mathematical utility functions
/// </summary>
internal static class MathFunctions
{
    /// <summary>
    /// Rounds a floating-point value to the nearest integer using the "round half up" strategy.
    /// This method adds 0.5 to the input value and then takes the floor.
    /// </summary>
    /// <typeparam name="T">A floating-point type that implements IFloatingPointIeee754</typeparam>
    /// <param name="x">The value to round</param>
    /// <returns>The rounded value as the same floating-point type</returns>
    public static T Round<T>(T x) where T : struct, IFloatingPointIeee754<T> => T.Floor(x + T.One / (T.One + T.One));
    
    /// <summary>
    /// Computes the standard sigmoid function: 1 / (1 + exp(-x)).
    /// The sigmoid function maps any real number to a value between 0 and 1,
    /// making it useful for probability calculations.
    /// </summary>
    /// <typeparam name="T">A floating-point type that implements IFloatingPointIeee754</typeparam>
    /// <param name="x">The input value</param>
    /// <returns>The sigmoid of x, a value between 0 and 1</returns>
    public static T StdSigmoid<T>(T x) where T : struct, IFloatingPointIeee754<T> => T.One / (T.One + T.Exp(-x));

    /// <summary>
    /// Computes the binary cross-entropy loss between predicted and target values.
    /// This is commonly used as a loss function in binary classification problems.
    /// The formula is: -[t * log(y + epsilon) + (1-t) * log(1-y + epsilon)]
    /// where epsilon is added for numerical stability to avoid log(0).
    /// </summary>
    /// <typeparam name="T">A floating-point type that implements IFloatingPointIeee754</typeparam>
    /// <param name="y">The predicted value (typically between 0 and 1)</param>
    /// <param name="t">The target value (typically 0 or 1)</param>
    /// <returns>The binary cross-entropy loss value</returns>
    public static T BinaryCrossEntropy<T>(T y, T t) where T : struct, IFloatingPointIeee754<T>
        => -(t * T.Log(y + T.Epsilon)
        + (T.One - t) * T.Log(T.One - y + T.Epsilon));

    /// <summary>
    /// Computes the floor of log base 2 of the given value.
    /// This is equivalent to finding the position of the most significant bit.
    /// Uses bit manipulation for efficient computation: 63 - leading_zero_count(x).
    /// </summary>
    /// <param name="x">The input value (must be greater than 0)</param>
    /// <returns>The floor of log base 2 of x</returns>
    /// <example>
    /// FloorLog2(8) returns 3, FloorLog2(15) returns 3, FloorLog2(16) returns 4
    /// </example>
    public static int FloorLog2(ulong x) => 63 - BitOperations.LeadingZeroCount(x);
}