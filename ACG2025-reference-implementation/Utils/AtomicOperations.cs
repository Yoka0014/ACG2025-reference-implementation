namespace ACG2025_reference_implementation.Utils;

using System;
using System.Numerics;
using System.Threading;
using System.Runtime.CompilerServices;

internal static class AtomicOperations
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Add<T>(ref T target, T arg) where T : struct, IFloatingPointIeee754<T>
    {
        if (T.IsNaN(target) || T.IsNaN(arg))
            return;

        if (typeof(T) == typeof(float))
        {
            ref var targetBits = ref Unsafe.As<T, uint>(ref target);
            T expected;
            uint expectedBits;
            T sum;
            do
            {
                expected = target;
                expectedBits = Unsafe.As<T, uint>(ref expected);
                sum = expected + arg;
            }
            while (Interlocked.CompareExchange(ref targetBits, Unsafe.As<T, uint>(ref sum), expectedBits) != expectedBits);
        }
        else if (typeof(T) == typeof(double))
        {
            ref var targetBits = ref Unsafe.As<T, ulong>(ref target);
            T expected;
            ulong expectedBits;
            T sum;
            do
            {
                expected = target;
                expectedBits = Unsafe.As<T, ulong>(ref expected);
                sum = expected + arg;
            }
            while (Interlocked.CompareExchange(ref targetBits, Unsafe.As<T, ulong>(ref sum), expectedBits) != expectedBits);
        }
        else
            throw new NotSupportedException();
    }
}