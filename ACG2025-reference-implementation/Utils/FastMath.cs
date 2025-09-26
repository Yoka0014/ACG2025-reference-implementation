namespace ACG2025_reference_implementation.Utils;

using System.Runtime.CompilerServices;

internal static class FastMath
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Exp2(float x)
    {
        int exp;
        if (x < 0)
        {
            if (x < -126)
                return 0.0f;
            exp = (int)(x - 1);
        }
        else
            exp = (int)x;

        float output = x - exp;
        output = 1.0f + output * (0.6602339f + 0.33976606f * output);
        var tmp = Unsafe.As<float, int>(ref output);
        tmp += (int)((uint)exp << 23);
        return Unsafe.As<int, float>(ref tmp);
    }

    public static float Exp(float x) => Exp2(1.442695040f * x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Log2(float x)
    {
        var tmp = Unsafe.As<float, uint>(ref x);
        var expb = tmp >> 23;
        tmp = (tmp & 0x7fffff) | (0x7f << 23);
        var output = Unsafe.As<uint, float>(ref tmp);
        output -= 1.0f;
        return output * (1.3465552f - 0.34655523f * output) - 127 + expb;
    }

    public static float Log(float x) => 0.6931471805599453f * Log2(x);
}