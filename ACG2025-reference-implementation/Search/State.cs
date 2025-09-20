namespace ACG2025_reference_implementation.Search;

using System;
using System.Runtime.CompilerServices;

using ACG2025_reference_implementation.Reversi;
using ACG2025_reference_implementation.NTupleSystem;
using ACG2025_reference_implementation.Evaluation;

internal struct State
{
    public Position Position;
    public FeatureVector FeatureVector;

    public State(NTupleManager nTupleManager) : this(new Position(), nTupleManager) { }

    public State(Position pos, NTupleManager nTuplesManager)
    {
        Position = pos;
        FeatureVector = new FeatureVector(nTuplesManager);
        FeatureVector.Init(ref pos);
    }

    public readonly void CopyTo(ref State dest)
    {
        dest.Position = Position;
        FeatureVector.CopyTo(dest.FeatureVector);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Update(ref Move move)
    {
        this.Position.Update(ref move);
        this.FeatureVector.Update(ref move);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Pass()
    {
        this.Position.Pass();
        this.FeatureVector.Pass();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Undo(ref Move move)
    {
        this.Position.Undo(ref move);
        this.FeatureVector.Undo(ref move);
    }
}