using System;
using System.Numerics;

namespace ACG2025_reference_implementation.Utils;

internal static class SpanExtensions
{
    public static T Sum<T>(this ReadOnlySpan<T> span) where T : struct, INumber<T>
    {
        var sum = T.Zero;
        foreach (var n in span)
            sum += n;
        return sum;
    }
}