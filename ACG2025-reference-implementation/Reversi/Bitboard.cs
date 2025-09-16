namespace ACG2025_reference_implementation.Reversi;

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics.Arm;

using ACG2025_reference_implementation.Utils;

using static Constants;

/// <summary>
/// Struct for managing discs on the board using bitboards.
/// Black and white discs are each managed as a 64-bit integer.
/// </summary>
internal struct Bitboard
{
    public ulong Player { get; private set; }
    public ulong Opponent { get; private set; }
    public readonly ulong Discs => Player | Opponent;
    public readonly ulong Empties => ~Discs;
    public readonly int PlayerDiscCount => BitOperations.PopCount(Player);
    public readonly int OpponentDiscCount => BitOperations.PopCount(Opponent);
    public readonly int DiscCount => BitOperations.PopCount(Discs);
    public readonly int EmptyCellsCount => BitOperations.PopCount(Empties);

    /// <summary>
    /// Random number table to compute Zobrist's hash.
    /// If CPU does not have any instructions to compute CRC32, the bitboard's hash code is computed using Zobrist's hash algorithm.
    /// If CPU has the CRC32 instruction, this table remains empty.
    /// </summary>
    static readonly ulong[][,] _HashRandTable = [];

    static Bitboard()
    {
        if (!Sse42.IsSupported && !Sse42.X64.IsSupported)
        {
            _HashRandTable = new ulong[2][,];
            for (var player = 0; player < 2; player++)
            {
                // Random numbers are assigned to each bit array in each row.
                // If the board size is 8, there can be 2 ^ 8 bit arrays for each row.
                // So 8 * 2 ^ 8 random numbers are needed for each player.
                var table = _HashRandTable[player] = new ulong[BoardSize, 2 << BoardSize];
                for (var j = 0; j < table.GetLength(0); j++)
                    for (var i = 0; i < table.GetLength(1); i++)
                        table[j, i] = (ulong)Random.Shared.NextInt64();
            }
        }
    }

    public Bitboard(ulong player, ulong opponent) => (Player, Opponent) = (player, opponent);

    public static bool operator ==(Bitboard left, Bitboard right)
        => left.Player == right.Player && left.Opponent == right.Opponent;

    public static bool operator !=(Bitboard left, Bitboard right)
        => !(left == right);

    // This method is only for suppressing a warning.
    public override readonly bool Equals(object? obj)
        => (obj is Bitboard bb) && this == bb;

    // This method is only for suppressing a warning.
    public override readonly int GetHashCode()
        => (int)ComputeHashCode();

    public void MirrorHorizontal()
    {
        Player = MirrorHorizontal(Player);
        Opponent = MirrorHorizontal(Opponent);
    }

    public void Rotate90Clockwise()
    {
        Player = Rotate90Clockwise(Player);
        Opponent = Rotate90Clockwise(Opponent);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Player GetSquareOwnerAt(BoardCoordinate coord)
    {
        var c = (int)coord;
        return (Player)(2 - 2 * ((Player >> c) & 1) - ((Opponent >> c) & 1));
    }

    public readonly ulong CalcPlayerMobility()
        => CalcMobility(Player, Opponent);

    public readonly ulong CalcOpponentMobility()
        => CalcMobility(Opponent, Player);

    public readonly ulong CalcFlip(BoardCoordinate coord)
        => CalcFlip(Player, Opponent, coord);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void PutPlayerDiscAt(BoardCoordinate coord)
    {
        var bit = 1UL << (int)coord;
        Player |= bit;
        if ((Opponent & bit) != 0)
            Opponent ^= bit;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void PutOpponentDiscAt(BoardCoordinate coord)
    {
        ulong bit = 1UL << (int)coord;
        Opponent |= bit;
        if ((Player & bit) != 0)
            Player ^= bit;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void RemoveDiscAt(BoardCoordinate coord)
    {
        ulong bit = 1UL << (int)coord;
        Player &= ~bit;
        Opponent &= ~bit;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Update(BoardCoordinate coord, ulong flip)
    {
        ulong player = Player;
        Player = Opponent ^ flip;
        Opponent = player | (1UL << (int)coord | flip);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Undo(BoardCoordinate coord, ulong flip)
    {
        ulong player = Player ^ flip;
        Player = Opponent ^ (1UL << (int)coord | flip);
        Opponent = player | flip;
    }

    public void Swap() => (Player, Opponent) = (Opponent, Player);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly ulong ComputeHashCode()
    {
        if (!Sse42.IsSupported && !Crc32.IsSupported)
            return ComputeZobristHashCode();

        var hp = ComputeCrc32(0, Player);
        return (hp << 32) | ComputeCrc32((uint)hp, Opponent);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    readonly unsafe ulong ComputeZobristHashCode()
    {
        ulong[,] pRand = _HashRandTable[0];
        ulong[,] oRand = _HashRandTable[1];

        var hp = 0UL;
        var ho = 0UL;

        (ulong player, ulong opponent) = (Player, Opponent);
        var p = (byte*)&player;
        var o = (byte*)&opponent;
        for (var i = 0; i < BoardSize; i++)
        {
            hp ^= pRand[i, p[i]];
            ho ^= oRand[i, o[i]];
        }

        return hp ^ ho;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static ulong PackA1H8(ulong bits)
    {
        const ulong Mask = 0x0101010101010101UL;

        if (Bmi2.X64.IsSupported)
            return Bmi2.X64.ParallelBitExtract(bits, Mask);
        else
        {
            // Shifter = 2^56 + 2^49 + 2^42 + 2^35 + 2^28 + 2^21 + 2^14 + 2^7
            const ulong Shifter = 0x0102040810204080UL;
            return ((bits & Mask) * Shifter) >> 56;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static ulong PackH1H8(ulong bits)
    {
        const ulong Mask = 0x8080808080808080UL;

        if (Bmi2.X64.IsSupported)
            return Bmi2.X64.ParallelBitExtract(bits, Mask);
        else
        {
            // Shifter = 2^49 + 2^42 + 2^35 + 2^28 + 2^21 + 2^14 + 2^7 + 2^0
            const ulong Shifter = 0x0002040810204081UL;
            return ((bits & Mask) * Shifter) >> 56;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static ulong MirrorHorizontal(ulong bitboard)
    {
        bitboard = BitManipulations.DeltaSwap(bitboard, 0x5555555555555555, 1);
        bitboard = BitManipulations.DeltaSwap(bitboard, 0x3333333333333333, 2);
        return BitManipulations.DeltaSwap(bitboard, 0x0f0f0f0f0f0f0f0f, 4);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static ulong MirrorVertical(ulong bitboard)
    {
        bitboard = BitManipulations.DeltaSwap(bitboard, 0x00ff00ff00ff00ff, 8);
        bitboard = BitManipulations.DeltaSwap(bitboard, 0x0000ffff0000ffff, 16);
        return BitManipulations.DeltaSwap(bitboard, 0x00000000ffffffff, 32);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static ulong MirrorDiagA1H8(ulong bitboard)
    {
        bitboard = BitManipulations.DeltaSwap(bitboard, 0x00aa00aa00aa00aa, 7);
        bitboard = BitManipulations.DeltaSwap(bitboard, 0x0000cccc0000cccc, 14);
        return BitManipulations.DeltaSwap(bitboard, 0x00000000f0f0f0f0, 28);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static ulong MirrorDiagA8H1(ulong bitboard)
    {
        bitboard = BitManipulations.DeltaSwap(bitboard, 0x0055005500550055UL, 9);
        bitboard = BitManipulations.DeltaSwap(bitboard, 0x0000333300003333UL, 18);
        return BitManipulations.DeltaSwap(bitboard, 0x000000000f0f0f0fUL, 36);
    }

    static ulong Rotate90Clockwise(ulong bitboard) => MirrorHorizontal(MirrorDiagA1H8(bitboard));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static ulong ComputeCrc32(uint crc, ulong data)
    {
        if (Crc32.IsSupported)
        {
            if (Crc32.Arm64.IsSupported)
                return Crc32.Arm64.ComputeCrc32(crc, data);
            else
                return Crc32.ComputeCrc32(Crc32.ComputeCrc32(crc, (uint)data), (uint)(data >> 32));
        }

        if (Sse42.X64.IsSupported)
            return Sse42.X64.Crc32(crc, data);
        return Sse42.Crc32(Sse42.Crc32(crc, (uint)data), (uint)(data >> 32));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static ulong CalcMobility(ulong player, ulong opponent)
    {
        if (Avx2.IsSupported)
            return CalcMobility_AVX2(player, opponent);

        if (Sse2.IsSupported)
            return CalcMobility_SSE(player, opponent);

        return CalcMobility_General(player, opponent);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static ulong CalcFlip(ulong player, ulong opponent, BoardCoordinate coord)
    {
        if (Avx2.IsSupported)
            return CalcFlip_AVX2(player, opponent, coord);

        if (Sse2.IsSupported)
            return CalcFlip_SSE(player, opponent, coord);

        return CalcFlip_General(player, opponent, coord);
    }

    /*
    * The algorithms for computing mobility and flipping discs are based on the following website:
    * ref: http://www.amy.hi-ho.ne.jp/okuhara/bitboard.htm (Japanese)
    */
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static ulong CalcMobility_AVX2(ulong p, ulong o)
    {
        var shift = Vector256.Create(1UL, 8UL, 9UL, 7UL);
        var shift2 = Vector256.Create(2UL, 16UL, 18UL, 14UL);
        var flipMask = Vector256.Create(0x7e7e7e7e7e7e7e7eUL, 0xffffffffffffffffUL, 0x7e7e7e7e7e7e7e7eUL, 0x7e7e7e7e7e7e7e7eUL);

        var p4 = Avx2.BroadcastScalarToVector256(Sse2.X64.ConvertScalarToVector128UInt64(p));
        var maskedO4 = Avx2.And(Avx2.BroadcastScalarToVector256(Sse2.X64.ConvertScalarToVector128UInt64(o)), flipMask);
        var prefixLeft = Avx2.And(maskedO4, Avx2.ShiftLeftLogicalVariable(maskedO4, shift));
        var prefixRight = Avx2.ShiftRightLogicalVariable(prefixLeft, shift);

        var flipLeft = Avx2.And(maskedO4, Avx2.ShiftLeftLogicalVariable(p4, shift));
        var flipRight = Avx2.And(maskedO4, Avx2.ShiftRightLogicalVariable(p4, shift));
        flipLeft = Avx2.Or(flipLeft, Avx2.And(maskedO4, Avx2.ShiftLeftLogicalVariable(flipLeft, shift)));
        flipRight = Avx2.Or(flipRight, Avx2.And(maskedO4, Avx2.ShiftRightLogicalVariable(flipRight, shift)));
        flipLeft = Avx2.Or(flipLeft, Avx2.And(prefixLeft, Avx2.ShiftLeftLogicalVariable(flipLeft, shift2)));
        flipRight = Avx2.Or(flipRight, Avx2.And(prefixRight, Avx2.ShiftRightLogicalVariable(flipRight, shift2)));
        flipLeft = Avx2.Or(flipLeft, Avx2.And(prefixLeft, Avx2.ShiftLeftLogicalVariable(flipLeft, shift2)));
        flipRight = Avx2.Or(flipRight, Avx2.And(prefixRight, Avx2.ShiftRightLogicalVariable(flipRight, shift2)));

        var mobility4 = Avx2.ShiftLeftLogicalVariable(flipLeft, shift);
        mobility4 = Avx2.Or(mobility4, Avx2.ShiftRightLogicalVariable(flipRight, shift));
        var mobility2 = Sse2.Or(Avx2.ExtractVector128(mobility4, 0), Avx2.ExtractVector128(mobility4, 1));
        mobility2 = Sse2.Or(mobility2, Sse2.UnpackHigh(mobility2, mobility2));
        return Sse2.X64.ConvertToUInt64(mobility2) & ~(p | o);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static ulong CalcMobility_SSE(ulong p, ulong o)
    {
        var maskedO = o & 0x7e7e7e7e7e7e7e7eUL;
        var p2 = Vector128.Create(p, BitManipulations.ByteSwap(p));   // byte swap = vertical mirror
        var maskedO2 = Vector128.Create(maskedO, BitManipulations.ByteSwap(maskedO));
        var prefix = Sse2.And(maskedO2, Sse2.ShiftLeftLogical(maskedO2, 7));
        var prefix1 = maskedO & (maskedO << 1);
        var prefix8 = o & (o << 8);

        var flip = Sse2.And(maskedO2, Sse2.ShiftLeftLogical(p2, 7));
        var flip1 = maskedO & (p << 1);
        var flip8 = o & (p << 8);
        flip = Sse2.Or(flip, Sse2.And(maskedO2, Sse2.ShiftLeftLogical(flip, 7)));
        flip1 |= maskedO & (flip1 << 1);
        flip8 |= o & (flip8 << 8);
        flip = Sse2.Or(flip, Sse2.And(prefix, Sse2.ShiftLeftLogical(flip, 14)));
        flip1 |= prefix1 & (flip1 << 2);
        flip8 |= prefix8 & (flip8 << 16);
        flip = Sse2.Or(flip, Sse2.And(prefix, Sse2.ShiftLeftLogical(flip, 14)));
        flip1 |= prefix1 & (flip1 << 2);
        flip8 |= prefix8 & (flip8 << 16);

        var mobility2 = Sse2.ShiftLeftLogical(flip, 7);
        var mobility = (flip1 << 1) | (flip8 << 8);

        prefix = Sse2.And(maskedO2, Sse2.ShiftLeftLogical(maskedO2, 9));
        prefix1 >>= 1;
        prefix8 >>= 8;
        flip = Sse2.And(maskedO2, Sse2.ShiftLeftLogical(p2, 9));
        flip1 = maskedO & (p >> 1);
        flip8 = o & (p >> 8);
        flip = Sse2.Or(flip, Sse2.And(maskedO2, Sse2.ShiftLeftLogical(flip, 9)));
        flip1 |= maskedO & (flip1 >> 1);
        flip8 |= o & (flip8 >> 8);
        flip = Sse2.Or(flip, Sse2.And(prefix, Sse2.ShiftLeftLogical(flip, 18)));
        flip1 |= prefix1 & (flip1 >> 2);
        flip8 |= prefix8 & (flip8 >> 16);
        flip = Sse2.Or(flip, Sse2.And(prefix, Sse2.ShiftLeftLogical(flip, 18)));
        flip1 |= prefix1 & (flip1 >> 2);
        flip8 |= prefix8 & (flip8 >> 16);
        mobility2 = Sse2.Or(mobility2, Sse2.ShiftLeftLogical(flip, 9));
        mobility |= (flip1 >> 1) | (flip8 >> 8);

        if (Sse2.X64.IsSupported)
            mobility |= Sse2.X64.ConvertToUInt64(mobility2) | BitManipulations.ByteSwap(Sse2.X64.ConvertToUInt64(Sse2.UnpackHigh(mobility2, mobility2)));
        else
            mobility |= mobility2.GetElement(0) | BitManipulations.ByteSwap(Sse2.UnpackHigh(mobility2, mobility2).GetElement(0));
        return mobility & ~(p | o);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static ulong CalcMobility_General(ulong p, ulong o)
    {
        var masked_o = o & 0x7e7e7e7e7e7e7e7eUL;

        // left
        ulong flip_horizontal = masked_o & (p << 1);
        ulong flip_diag_A1H8 = masked_o & (p << 9);
        ulong flip_diag_A8H1 = masked_o & (p << 7);
        ulong flip_vertical = o & (p << 8);

        flip_horizontal |= masked_o & (flip_horizontal << 1);
        flip_diag_A1H8 |= masked_o & (flip_diag_A1H8 << 9);
        flip_diag_A8H1 |= masked_o & (flip_diag_A8H1 << 7);
        flip_vertical |= o & (flip_vertical << 8);

        ulong prefix_horizontal = masked_o & (masked_o << 1);
        ulong prefix_diag_A1H8 = masked_o & (masked_o << 9);
        ulong prefix_diag_A8H1 = masked_o & (masked_o << 7);
        ulong prefix_vertical = o & (o << 8);

        flip_horizontal |= prefix_horizontal & (flip_horizontal << 2);
        flip_diag_A1H8 |= prefix_diag_A1H8 & (flip_diag_A1H8 << 18);
        flip_diag_A8H1 |= prefix_diag_A8H1 & (flip_diag_A8H1 << 14);
        flip_vertical |= prefix_vertical & (flip_vertical << 16);

        flip_horizontal |= prefix_horizontal & (flip_horizontal << 2);
        flip_diag_A1H8 |= prefix_diag_A1H8 & (flip_diag_A1H8 << 18);
        flip_diag_A8H1 |= prefix_diag_A8H1 & (flip_diag_A8H1 << 14);
        flip_vertical |= prefix_vertical & (flip_vertical << 16);

        ulong mobility = (flip_horizontal << 1) | (flip_diag_A1H8 << 9) | (flip_diag_A8H1 << 7) | (flip_vertical << 8);

        // right
        flip_horizontal = masked_o & (p >> 1);
        flip_diag_A1H8 = masked_o & (p >> 9);
        flip_diag_A8H1 = masked_o & (p >> 7);
        flip_vertical = o & (p >> 8);

        flip_horizontal |= masked_o & (flip_horizontal >> 1);
        flip_diag_A1H8 |= masked_o & (flip_diag_A1H8 >> 9);
        flip_diag_A8H1 |= masked_o & (flip_diag_A8H1 >> 7);
        flip_vertical |= o & (flip_vertical >> 8);

        prefix_horizontal = masked_o & (masked_o >> 1);
        prefix_diag_A1H8 = masked_o & (masked_o >> 9);
        prefix_diag_A8H1 = masked_o & (masked_o >> 7);
        prefix_vertical = o & (o >> 8);

        flip_horizontal |= prefix_horizontal & (flip_horizontal >> 2);
        flip_diag_A1H8 |= prefix_diag_A1H8 & (flip_diag_A1H8 >> 18);
        flip_diag_A8H1 |= prefix_diag_A8H1 & (flip_diag_A8H1 >> 14);
        flip_vertical |= prefix_vertical & (flip_vertical >> 16);

        flip_horizontal |= prefix_horizontal & (flip_horizontal >> 2);
        flip_diag_A1H8 |= prefix_diag_A1H8 & (flip_diag_A1H8 >> 18);
        flip_diag_A8H1 |= prefix_diag_A8H1 & (flip_diag_A8H1 >> 14);
        flip_vertical |= prefix_vertical & (flip_vertical >> 16);

        mobility |= (flip_horizontal >> 1) | (flip_diag_A1H8 >> 9) | (flip_diag_A8H1 >> 7) | (flip_vertical >> 8);
        return mobility & ~(p | o);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static ulong CalcFlip_AVX2(ulong p, ulong o, BoardCoordinate coord)
    {
        var shift = Vector256.Create(1UL, 8UL, 9UL, 7UL);
        var shift2 = Vector256.Create(2UL, 16UL, 18UL, 14UL);
        var flipMask = Vector256.Create(0x7e7e7e7e7e7e7e7eUL, 0xffffffffffffffffUL, 0x7e7e7e7e7e7e7e7eUL, 0x7e7e7e7e7e7e7e7eUL);

        ulong x = 1UL << (int)coord;
        var x4 = Avx2.BroadcastScalarToVector256(Sse2.X64.ConvertScalarToVector128UInt64(x));
        var p4 = Avx2.BroadcastScalarToVector256(Sse2.X64.ConvertScalarToVector128UInt64(p));
        var maskedO4 = Avx2.And(Avx2.BroadcastScalarToVector256(Sse2.X64.ConvertScalarToVector128UInt64(o)), flipMask);
        var prefixLeft = Avx2.And(maskedO4, Avx2.ShiftLeftLogicalVariable(maskedO4, shift));
        var prefixRight = Avx2.ShiftRightLogicalVariable(prefixLeft, shift);

        var flipLeft = Avx2.And(Avx2.ShiftLeftLogicalVariable(x4, shift), maskedO4);
        var flipRight = Avx2.And(Avx2.ShiftRightLogicalVariable(x4, shift), maskedO4);
        flipLeft = Avx2.Or(flipLeft, Avx2.And(maskedO4, Avx2.ShiftLeftLogicalVariable(flipLeft, shift)));
        flipRight = Avx2.Or(flipRight, Avx2.And(maskedO4, Avx2.ShiftRightLogicalVariable(flipRight, shift)));
        flipLeft = Avx2.Or(flipLeft, Avx2.And(prefixLeft, Avx2.ShiftLeftLogicalVariable(flipLeft, shift2)));
        flipRight = Avx2.Or(flipRight, Avx2.And(prefixRight, Avx2.ShiftRightLogicalVariable(flipRight, shift2)));
        flipLeft = Avx2.Or(flipLeft, Avx2.And(prefixLeft, Avx2.ShiftLeftLogicalVariable(flipLeft, shift2)));
        flipRight = Avx2.Or(flipRight, Avx2.And(prefixRight, Avx2.ShiftRightLogicalVariable(flipRight, shift2)));

        var outflankLeft = Avx2.And(p4, Avx2.ShiftLeftLogicalVariable(flipLeft, shift));
        var outflankRight = Avx2.And(p4, Avx2.ShiftRightLogicalVariable(flipRight, shift));
        flipLeft = Avx2.AndNot(Avx2.CompareEqual(outflankLeft, Vector256<ulong>.Zero), flipLeft);
        flipRight = Avx2.AndNot(Avx2.CompareEqual(outflankRight, Vector256<ulong>.Zero), flipRight);
        var flip4 = Avx2.Or(flipLeft, flipRight);
        var flip2 = Sse2.Or(Avx2.ExtractVector128(flip4, 0), Avx2.ExtractVector128(flip4, 1));
        flip2 = Sse2.Or(flip2, Sse2.UnpackHigh(flip2, flip2));
        return Sse2.X64.ConvertToUInt64(flip2);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static ulong CalcFlip_SSE(ulong p, ulong o, BoardCoordinate coord)
    {
        ulong x = 1UL << (int)coord;
        var maskedO = o & 0x7e7e7e7e7e7e7e7eUL;
        var x2 = Vector128.Create(x, BitManipulations.ByteSwap(x));   // byte swap = vertical mirror
        var p2 = Vector128.Create(p, BitManipulations.ByteSwap(p));
        var maskedO2 = Vector128.Create(maskedO, BitManipulations.ByteSwap(maskedO));
        var prefix = Sse2.And(maskedO2, Sse2.ShiftLeftLogical(maskedO2, 7));
        var prefix1 = maskedO & (maskedO << 1);
        var prefix8 = o & (o << 8);

        var flip7 = Sse2.And(maskedO2, Sse2.ShiftLeftLogical(x2, 7));
        var flip1Left = maskedO & (x << 1);
        var flip8Left = o & (x << 8);
        flip7 = Sse2.Or(flip7, Sse2.And(maskedO2, Sse2.ShiftLeftLogical(flip7, 7)));
        flip1Left |= maskedO & (flip1Left << 1);
        flip8Left |= o & (flip8Left << 8);
        flip7 = Sse2.Or(flip7, Sse2.And(prefix, Sse2.ShiftLeftLogical(flip7, 14)));
        flip1Left |= prefix1 & (flip1Left << 2);
        flip8Left |= prefix8 & (flip8Left << 16);
        flip7 = Sse2.Or(flip7, Sse2.And(prefix, Sse2.ShiftLeftLogical(flip7, 14)));
        flip1Left |= prefix1 & (flip1Left << 2);
        flip8Left |= prefix8 & (flip8Left << 16);

        prefix = Sse2.And(maskedO2, Sse2.ShiftLeftLogical(maskedO2, 9));
        prefix1 >>= 1;
        prefix8 >>= 8;

        var flip9 = Sse2.And(maskedO2, Sse2.ShiftLeftLogical(x2, 9));
        var flip1Right = maskedO & (x >> 1);
        var flip8Right = o & (x >> 8);
        flip9 = Sse2.Or(flip9, Sse2.And(maskedO2, Sse2.ShiftLeftLogical(flip9, 9)));
        flip1Right |= maskedO & (flip1Right >> 1);
        flip8Right |= o & (flip8Right >> 8);
        flip9 = Sse2.Or(flip9, Sse2.And(prefix, Sse2.ShiftLeftLogical(flip9, 18)));
        flip1Right |= prefix1 & (flip1Right >> 2);
        flip8Right |= prefix8 & (flip8Right >> 16);
        flip9 = Sse2.Or(flip9, Sse2.And(prefix, Sse2.ShiftLeftLogical(flip9, 18)));
        flip1Right |= prefix1 & (flip1Right >> 2);
        flip8Right |= prefix8 & (flip8Right >> 16);

        var outflank7 = Sse2.And(p2, Sse2.ShiftLeftLogical(flip7, 7));
        var outflankLeft1 = p & (flip1Left << 1);
        var outflankLeft8 = p & (flip8Left << 8);
        var outflank9 = Sse2.And(p2, Sse2.ShiftLeftLogical(flip9, 9));
        var outflankRight1 = p & (flip1Right >> 1);
        var outflankRight8 = p & (flip8Right >> 8);

        if (Sse41.IsSupported)
        {
            flip7 = Sse2.AndNot(Sse41.CompareEqual(outflank7, Vector128<ulong>.Zero), flip7);
            flip9 = Sse2.AndNot(Sse41.CompareEqual(outflank9, Vector128<ulong>.Zero), flip9);
        }
        else
        {
            flip7 = Sse2.And(Sse2.CompareNotEqual(outflank7.AsDouble(), Vector128<ulong>.Zero.AsDouble()).AsUInt64(), flip7);
            flip9 = Sse2.And(Sse2.CompareNotEqual(outflank9.AsDouble(), Vector128<ulong>.Zero.AsDouble()).AsUInt64(), flip9);
        }

        if (outflankLeft1 == 0)
            flip1Left = 0UL;
        if (outflankLeft8 == 0)
            flip8Left = 0UL;
        if (outflankRight1 == 0)
            flip1Right = 0UL;
        if (outflankRight8 == 0)
            flip8Right = 0UL;

        var flippedDiscs2 = Sse2.Or(flip7, flip9);
        var flippedDiscs = flip1Left | flip8Left | flip1Right | flip8Right;
        if (Sse2.X64.IsSupported)
            flippedDiscs |= Sse2.X64.ConvertToUInt64(flippedDiscs2)
                            | BitManipulations.ByteSwap(Sse2.X64.ConvertToUInt64(Sse2.UnpackHigh(flippedDiscs2, flippedDiscs2)));
        else
            flippedDiscs |= flippedDiscs2.GetElement(0) | BitManipulations.ByteSwap(Sse2.UnpackHigh(flippedDiscs2, flippedDiscs2).GetElement(0));
        return flippedDiscs;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static ulong CalcFlip_General(ulong p, ulong o, BoardCoordinate coord)
    {
        ulong x = 1UL << (int)coord;
        var masked_o = o & 0x7e7e7e7e7e7e7e7eUL;

        // left
        ulong flipped_horizontal = masked_o & (x << 1);
        ulong flipped_diag_A1H8 = masked_o & (x << 9);
        ulong flipped_diag_A8H1 = masked_o & (x << 7);
        ulong flipped_vertical = o & (x << 8);

        flipped_horizontal |= masked_o & (flipped_horizontal << 1);
        flipped_diag_A1H8 |= masked_o & (flipped_diag_A1H8 << 9);
        flipped_diag_A8H1 |= masked_o & (flipped_diag_A8H1 << 7);
        flipped_vertical |= o & (flipped_vertical << 8);

        ulong prefix_horizontal = masked_o & (masked_o << 1);
        ulong prefix_diag_A1H8 = masked_o & (masked_o << 9);
        ulong prefix_diag_A8H1 = masked_o & (masked_o << 7);
        ulong prefix_vertical = o & (o << 8);

        flipped_horizontal |= prefix_horizontal & (flipped_horizontal << 2);
        flipped_diag_A1H8 |= prefix_diag_A1H8 & (flipped_diag_A1H8 << 18);
        flipped_diag_A8H1 |= prefix_diag_A8H1 & (flipped_diag_A8H1 << 14);
        flipped_vertical |= prefix_vertical & (flipped_vertical << 16);

        flipped_horizontal |= prefix_horizontal & (flipped_horizontal << 2);
        flipped_diag_A1H8 |= prefix_diag_A1H8 & (flipped_diag_A1H8 << 18);
        flipped_diag_A8H1 |= prefix_diag_A8H1 & (flipped_diag_A8H1 << 14);
        flipped_vertical |= prefix_vertical & (flipped_vertical << 16);

        ulong outflank_horizontal = p & (flipped_horizontal << 1);
        ulong outflank_diag_A1H8 = p & (flipped_diag_A1H8 << 9);
        ulong outflank_diag_A8H1 = p & (flipped_diag_A8H1 << 7);
        ulong outflank_vertical = p & (flipped_vertical << 8);

        if (outflank_horizontal == 0)
            flipped_horizontal = 0UL;

        if (outflank_diag_A1H8 == 0)
            flipped_diag_A1H8 = 0UL;

        if (outflank_diag_A8H1 == 0)
            flipped_diag_A8H1 = 0UL;

        if (outflank_vertical == 0)
            flipped_vertical = 0UL;

        ulong flipped = flipped_horizontal | flipped_diag_A1H8 | flipped_diag_A8H1 | flipped_vertical;

        // right
        flipped_horizontal = masked_o & (x >> 1);
        flipped_diag_A1H8 = masked_o & (x >> 9);
        flipped_diag_A8H1 = masked_o & (x >> 7);
        flipped_vertical = o & (x >> 8);

        flipped_horizontal |= masked_o & (flipped_horizontal >> 1);
        flipped_diag_A1H8 |= masked_o & (flipped_diag_A1H8 >> 9);
        flipped_diag_A8H1 |= masked_o & (flipped_diag_A8H1 >> 7);
        flipped_vertical |= o & (flipped_vertical >> 8);

        prefix_horizontal = masked_o & (masked_o >> 1);
        prefix_diag_A1H8 = masked_o & (masked_o >> 9);
        prefix_diag_A8H1 = masked_o & (masked_o >> 7);
        prefix_vertical = o & (o >> 8);

        flipped_horizontal |= prefix_horizontal & (flipped_horizontal >> 2);
        flipped_diag_A1H8 |= prefix_diag_A1H8 & (flipped_diag_A1H8 >> 18);
        flipped_diag_A8H1 |= prefix_diag_A8H1 & (flipped_diag_A8H1 >> 14);
        flipped_vertical |= prefix_vertical & (flipped_vertical >> 16);

        flipped_horizontal |= prefix_horizontal & (flipped_horizontal >> 2);
        flipped_diag_A1H8 |= prefix_diag_A1H8 & (flipped_diag_A1H8 >> 18);
        flipped_diag_A8H1 |= prefix_diag_A8H1 & (flipped_diag_A8H1 >> 14);
        flipped_vertical |= prefix_vertical & (flipped_vertical >> 16);

        outflank_horizontal = p & (flipped_horizontal >> 1);
        outflank_diag_A1H8 = p & (flipped_diag_A1H8 >> 9);
        outflank_diag_A8H1 = p & (flipped_diag_A8H1 >> 7);
        outflank_vertical = p & (flipped_vertical >> 8);

        if (outflank_horizontal == 0UL)
            flipped_horizontal = 0UL;

        if (outflank_diag_A1H8 == 0UL)
            flipped_diag_A1H8 = 0UL;

        if (outflank_diag_A8H1 == 0UL)
            flipped_diag_A8H1 = 0UL;

        if (outflank_vertical == 0UL)
            flipped_vertical = 0UL;

        return flipped | flipped_horizontal | flipped_diag_A1H8 | flipped_diag_A8H1 | flipped_vertical;
    }
}