namespace ACG2025_reference_implementation.Utils;

/// <summary>
/// A structure that manages time control information based on the Canadian system.
/// 
/// About Canadian Time System:
/// 1. Initially, main time (MainTimeMs) is given to the player
/// 2. When main time is exhausted, the game enters the byoyomi phase
/// 3. In byoyomi phase, a specified number of moves (ByoyomiStones) 
///    must be made within the specified time (ByoyomiMs)
/// 4. If the required number of moves cannot be made within byoyomi time, it results in a time forfeit
/// 5. If IncrementMs is set, time is added after each move (Fischer time element)
/// 
/// Example: 30 minutes main time + 30 seconds for 5 moves byoyomi
/// - The first 30 minutes can be used freely
/// - After 30 minutes, 5 moves must be made within 30 seconds
/// - After completing 5 moves, a new 30-second 5-move byoyomi period begins
/// </summary>
internal struct TimeControl
{
    /// <summary>
    /// Main time in milliseconds.
    /// The basic thinking time available to the player.
    /// </summary>
    public int MainTimeMs { get; set; }

    /// <summary>
    /// Byoyomi time in milliseconds.
    /// The time limit per move after main time is exhausted.
    /// </summary>
    public int ByoyomiMs { get; set; }

    /// <summary>
    /// Increment time in milliseconds.
    /// Time added after each move (Fischer system).
    /// </summary>
    public int IncrementMs { get; set; }

    /// <summary>
    /// Number of stones in byoyomi.
    /// The number of moves that must be made within byoyomi time.
    /// </summary>
    public int ByoyomiStones { get; set; }
}