namespace ACG2025_reference_implementation.Utils;

using System;
using System.IO;

internal static class StreamExtensions
{
    public static int Read(this Stream stream, byte[] buffer, int offset, int count, bool swapBytes)
        => stream.Read(buffer.AsSpan(offset, count), swapBytes);

    public static int Read(this Stream stream, Span<byte> buffer, bool swapBytes)
    {
        var ret = stream.Read(buffer);
        if (swapBytes)
            SwapBytes(buffer);
        return ret;
    }

    public static void Write(this Stream stream, byte[] buffer, int offset, int count, bool swapBytes)
        => stream.Write(buffer.AsSpan(offset, count), swapBytes);

    public static void Write(this Stream stream, Span<byte> buffer, bool swapBytes)
    {
        if (swapBytes)
            SwapBytes(buffer);
        stream.Write(buffer);
    }

    static void SwapBytes(Span<byte> buffer)
    {
        for (var i = 0; i < buffer.Length / 2; i++)
            (buffer[i], buffer[buffer.Length - i - 1]) = (buffer[buffer.Length - i - 1], buffer[i]);
    }
}