using System.Numerics;

namespace ACG2025_reference_implementation.Utils;

internal static class MathFunctions
{
    public static T Round<T>(T x) where T : struct, IFloatingPointIeee754<T> => T.Floor(x + T.One / (T.One + T.One));
    public static T StdSigmoid<T>(T x) where T : struct, IFloatingPointIeee754<T> => T.One / (T.One + T.Exp(-x));
}