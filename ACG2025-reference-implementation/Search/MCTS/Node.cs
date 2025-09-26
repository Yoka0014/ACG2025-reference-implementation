namespace ACG2025_reference_implementation.Search.MCTS;

using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;

using ACG2025_reference_implementation.Reversi;

internal enum EdgeLabel
{
    NotProved = 0x00,
    Proved = 0xf0,
    Win = Proved | PUCTConstants.OutcomeWin,
    Loss = Proved | PUCTConstants.OutcomeLoss,
    Draw = Proved | PUCTConstants.OutcomeDraw
}

internal struct Edge
{
    public Move Move;
    public Half PriorProb;
    public Half Value;
    public uint VisitCount;
    public double ValueSum;
    public EdgeLabel Label;

    public readonly double ExpectedReward => ValueSum / VisitCount;
    public readonly bool IsProved => (Label & EdgeLabel.Proved) != 0;
    public readonly bool IsWin => Label == EdgeLabel.Win;
    public readonly bool IsLoss => Label == EdgeLabel.Loss;
    public readonly bool IsDraw => Label == EdgeLabel.Draw;

    public readonly bool PriorTo(ref Edge edge)
    {
        if (VisitCount == 0)
            return false;

        var diff = (long)VisitCount - edge.VisitCount;
        if (diff != 0)
            return diff > 0;
        return ExpectedReward > edge.ExpectedReward;
    }
}

internal class Node
{
    public uint VisitCount;
    public Edge[]? Edges;
    public Node[]? ChildNodes;

    public bool IsExpanded => Edges is not null;
    public bool ChildNodeWasInitialized => ChildNodes is not null;

    public double ExpectedReward
    {
        get
        {
            if (Edges is null)
                return double.NaN;

            var reward = 0.0;
            for (var i = 0; i < Edges.Length; i++)
            {
                Debug.Assert(VisitCount != 0);
                reward += Edges[i].ValueSum / VisitCount;
            }
            return reward;
        }
    }

    /// <summary>
    /// 指定されたインデックスに対応する子ノードのNodeオブジェクトを初期化する．
    /// </summary>
    /// <param name="idx"></param>
    /// <returns>生成されたNodeオブジェクト</returns>
    public Node CreateChildNode(int idx)
    {
        Debug.Assert(ChildNodes is not null);
        return ChildNodes[idx] = new Node();
    }

    /// <summary>
    /// 合法手数と同じ長さのNode配列を作成する．ただし，この時点では全ての要素はnullで初期化され，必要になった時にCreateChildNodeメソッドでNodeオブジェクトを生成する．
    /// </summary>
    /// <returns></returns>
    public Node[] InitChildNodes()
    {
        Debug.Assert(Edges is not null);
        return ChildNodes = new Node[Edges.Length];
    }

    /// <summary>
    /// 合法手数だけこのノードから辺を展開する．
    /// </summary>
    /// <param name="pos">このノードに対応する局面</param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Edge[] Expand(ref Position pos)
    {
        Span<Move> moves = stackalloc Move[Constants.MaxLegalMoves];
        var numMoves = pos.GetNextMoves(ref moves);

        if (numMoves == 0)
        {
            Edges = new Edge[1];
            Edges[0].Move.Coord = BoardCoordinate.Pass;
            return Edges;
        }

        Edges = new Edge[numMoves];
        for (var i = 0; i < Edges.Length; i++)
            Edges[i].Move = moves[i];

        return Edges;
    }
}