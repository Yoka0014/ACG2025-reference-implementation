namespace ACG2025_reference_implementation.Utils;

using System;

public static class RandomExtensions
{
    /// <summary>
    /// Samples an index according to the distribution specified by prob.
    /// </summary>
    /// <param name="rand">Random instance</param>
    /// <param name="prob">Array of probabilities (doesn't have to be normalized; will be auto-normalized)</param>
    /// <returns>Sampled index</returns>
    public static int Sample(this Random rand, ReadOnlySpan<double> prob)
    {
        if (prob.Length == 0)
            throw new ArgumentException("The \"prob\" array must not be empty.");

        double sum = 0;
        for (var i = 0; i < prob.Length; i++)
        {
            if (prob[i] < 0)
                throw new ArgumentException("Negative values are not allowed in the \"prob\" array.");

            sum += prob[i];
        }

        if (sum <= 0)
            throw new ArgumentException("The sum of \"prob\" array must be positive.");

        var r = rand.NextDouble() * sum;
        var cumulative = 0.0;
        for (var i = 0; i < prob.Length; i++)
        {
            cumulative += prob[i];
            if (r < cumulative)
                return i;
        }
        
        // Due to floating point error, return the last index as a fallback
        return prob.Length - 1;
    }
}