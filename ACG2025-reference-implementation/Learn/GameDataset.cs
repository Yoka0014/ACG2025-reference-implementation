global using GameDataset = System.Collections.Generic.List<ACG2025_reference_implementation.Learn.GameDatasetItem>;

namespace ACG2025_reference_implementation.Learn;

using System;
using System.Collections.Generic;

using ACG2025_reference_implementation.Reversi;

internal struct GameDatasetItem(Position rootPos, IEnumerable<Move> moves, IEnumerable<Half> evalScores, sbyte scoreFromBlack)
{
    public Position RootPos { get; } = rootPos;
    public sbyte ScoreFromBlack { get; } = scoreFromBlack;
    public Move[] Moves { get; set; } = [.. moves];
    public Half[] EvalScores { get; set; } = [.. evalScores];

    public GameDatasetItem(Position rootPos, IEnumerable<Move> moves, sbyte scoreFormBlack) : this(rootPos, moves, [], scoreFormBlack) { }

    public readonly int GetScoreFrom(DiscColor color)
    {
        if (color == DiscColor.Null)
            throw new ArgumentException($"{nameof(color)} cannot be {DiscColor.Null}.");
        return (color == DiscColor.Black) ? ScoreFromBlack : -ScoreFromBlack;
    }
}