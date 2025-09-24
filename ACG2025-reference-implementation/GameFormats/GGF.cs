namespace ACG2025_reference_implementation.GameFormats;

using System;
using System.Text;
using System.Linq;
using System.Collections.Generic;

using ACG2025_reference_implementation.Utils;
using ACG2025_reference_implementation.Reversi;

/// <summary>
/// Exception thrown when GGF parsing fails.
/// </summary>
/// <remarks>
/// This GGF parser is implemented specifically for communication with the NBoard protocol
/// and does not support the full GGF specification.
/// </remarks>
internal class GGFParserException(string message) : Exception(message) { }

/// <summary>
/// Represents the result of a GGF game.
/// </summary>
internal class GGFGameResult
{
    /// <summary>
    /// Gets or sets the score of the first player (Black).
    /// </summary>
    public double? FirstPlayerScore { get; set; }
    
    /// <summary>
    /// Gets or sets a value indicating whether the game ended by resignation.
    /// </summary>
    public bool IsResigned { get; set; }
    
    /// <summary>
    /// Gets or sets a value indicating whether the game ended by timeout.
    /// </summary>
    public bool IsTimeout { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether the game ended by mutual agreement.
    /// </summary>
    public bool IsMutual { get; set; }

    /// <summary>
    /// Gets a value indicating whether the game result is unknown.
    /// </summary>
    public bool IsUnknown => !FirstPlayerScore.HasValue;
}

/// <summary>
/// Represents a move in a GGF game.
/// </summary>
internal class GGFMove
{
    /// <summary>
    /// Gets or sets the color of the disc placed by this move.
    /// </summary>
    public DiscColor Color { get; set; }
    
    /// <summary>
    /// Gets or sets the board coordinate of this move.
    /// </summary>
    public BoardCoordinate Coord { get; set; }
    
    /// <summary>
    /// Gets or sets the evaluation score for this move.
    /// </summary>
    public double? EvalScore { get; set; }
    
    /// <summary>
    /// Gets or sets the time taken to make this move.
    /// </summary>
    public double? Time { get; set; }
}

/// <summary>
/// Represents a Reversi game in GGF (Generic Game Format) format.
/// </summary>
/// <remarks>
/// This GGF parser is implemented specifically for communication with the NBoard protocol
/// and does not support the full GGF specification.
/// </remarks>
internal class GGFReversiGame
{
    /// <summary>
    /// The delimiter that marks the start of a GGF game.
    /// </summary>
    const string GameStartDelimiter = "(;";
    
    /// <summary>
    /// The delimiter that marks the end of a GGF game.
    /// </summary>
    const string GameEndDelimiter = ";)";
    
    /// <summary>
    /// The character that marks the start of a property value.
    /// </summary>
    const char PropertyStartDelimiter = '[';
    
    /// <summary>
    /// The character that marks the end of a property value.
    /// </summary>
    const char PropertyEndDelimiter = ']';
    
    /// <summary>
    /// The game type identifier for Othello/Reversi in GGF format.
    /// </summary>
    const string GameType = "othello";

    /// <summary>
    /// Gets or sets the place where the game was played.
    /// </summary>
    public string Place { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the date when the game was played.
    /// According to the GGF specification, the format should be "year.month.day_hour:minute:second.zone".
    /// However, some GGF game records use UNIX time or other formats, so the date is stored as text.
    /// </summary>
    /// <seealso href="https://skatgame.net/mburo/ggsa/ggf"/>
    public string Date { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the name of the Black player.
    /// </summary>
    public string BlackPlayerName { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the name of the White player.
    /// </summary>
    public string WhitePlayerName { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the rating of the Black player.
    /// </summary>
    public double BlackPlayerRating { get; set; }
    
    /// <summary>
    /// Gets or sets the rating of the White player.
    /// </summary>
    public double WhitePlayerRating { get; set; }
    
    /// <summary>
    /// Gets or sets the time control settings for the Black player.
    /// </summary>
    public TimeControl BlackThinkingTime { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the time control settings for the White player.
    /// </summary>
    public TimeControl WhiteThinkingTime { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the game result.
    /// </summary>
    public GGFGameResult? GameResult { get; set; }
    
    /// <summary>
    /// Gets the list of moves played in the game.
    /// </summary>
    public List<GGFMove> Moves { get; } = [];

    /// <summary>
    /// The initial position of the game.
    /// </summary>
    Position _position;

    /// <summary>
    /// Initializes a new instance of the <see cref="GGFReversiGame"/> class by parsing a GGF string.
    /// </summary>
    /// <param name="ggfStr">The GGF string to parse.</param>
    /// <exception cref="GGFParserException">Thrown when the GGF string is invalid or cannot be parsed.</exception>
    public GGFReversiGame(string ggfStr)
    {
        var tokenizer = new Tokenizer { Input = ggfStr };

        if (!FindGameStartDelimiter(tokenizer))
            throw new GGFParserException($"GGF must start with \"{GameStartDelimiter}\"");

        ParseProperties(tokenizer);
    }

    /// <summary>
    /// Gets the initial position of the game.
    /// </summary>
    /// <returns>The initial position.</returns>
    public Position GetPosition() => _position;

    /// <summary>
    /// Parses the properties from the GGF string.
    /// </summary>
    /// <param name="tokenizer">The tokenizer used for parsing.</param>
    /// <exception cref="GGFParserException">Thrown when the GGF string contains invalid property syntax.</exception>
    void ParseProperties(Tokenizer tokenizer)
    {
        var sb = new StringBuilder();

        char ch;
        while(!tokenizer.IsEndOfString)
        {
            ch = tokenizer.ReadNextChar();
            if(ch == GameEndDelimiter[0])
            {
                if (tokenizer.ReadNextChar() == GameEndDelimiter[1])
                    return;
                throw new GGFParserException($"Unexpected token \"{GameEndDelimiter[0]}\". Maybe \"{GameEndDelimiter}\"?");
            }

            if (ch >= 'A' && ch <= 'Z')
            {
                sb.Clear();
                sb.Append(ch);

                while (!tokenizer.IsEndOfString)
                {
                    ch = tokenizer.ReadNextChar();
                    if (ch == PropertyStartDelimiter)
                        break;

                    if (ch < 'A' || ch > 'Z')
                        throw new GGFParserException($"The property name contains invalid character \'{ch}\'");

                    sb.Append(ch);
                }

                if (tokenizer.IsEndOfString)
                    throw new GGFParserException($"GGF must end with \"{GameEndDelimiter}\"");

                ParseProperty(sb.ToString(), tokenizer);
            }
        }
    }

    /// <summary>
    /// Parses a single property from the GGF string.
    /// </summary>
    /// <param name="propertyName">The name of the property to parse.</param>
    /// <param name="tokenizer">The tokenizer used for parsing.</param>
    /// <exception cref="GGFParserException">Thrown when property values are invalid or cannot be parsed.</exception>
    void ParseProperty(string propertyName, Tokenizer tokenizer)
    {
        var value = tokenizer.ReadTo(PropertyEndDelimiter);

        switch (propertyName) 
        {
            case "GM":
                var lvalue = value.ToLower();
                if (lvalue != GameType)
                    throw new GGFParserException($"Game \"{value}\" is not supported.");
                return;

            case "PC":
                Place = value;
                return;

            case "DT":
                Date = value;
                return;

            case "PB":
                BlackPlayerName = value;
                return;

            case "PW":
                WhitePlayerName = value;
                return;

            case "RB":
                if (!double.TryParse(value, out double blackRating))
                    throw new GGFParserException("The value of RB must be a real number.");
                BlackPlayerRating = blackRating;
                return;

            case "RW":
                if (!double.TryParse(value, out double whiteRating))
                    throw new GGFParserException("The value of RW must be a real number.");
                WhitePlayerRating = whiteRating;
                return;

            case "TI":
                BlackThinkingTime = ParseTime(value);
                WhiteThinkingTime = BlackThinkingTime;
                return;

            case "TB":
                BlackThinkingTime = ParseTime(value);
                return;

            case "TW":
                WhiteThinkingTime = ParseTime(value);
                return;

            case "RE":
                GameResult = ParseResult(value);
                return;

            case "BO":
                _position = ParsePosition(value);
                return;

            case "B":
                var bmove = ParseMove(DiscColor.Black, value);
                Moves.Add(bmove);
                return;

            case "W":
                var wmove = ParseMove(DiscColor.White, value);
                Moves.Add(wmove);
                return;
        }
    }

    /// <summary>
    /// Searches for the game start delimiter in the tokenizer input.
    /// </summary>
    /// <param name="tokenizer">The tokenizer to search in.</param>
    /// <returns><c>true</c> if the game start delimiter is found; otherwise, <c>false</c>.</returns>
    static bool FindGameStartDelimiter(Tokenizer tokenizer)
    {
        while (!tokenizer.IsEndOfString)
            if (tokenizer.ReadNextChar() == GameStartDelimiter[0] && tokenizer.ReadNextChar() == GameStartDelimiter[1])
                return true;
        return false;
    }

    /// <summary>
    /// Parses a time control string from GGF format.
    /// </summary>
    /// <param name="timeStr">The time string to parse in format "[main_time]/[increment_time]/[extension_time]".</param>
    /// <returns>A <see cref="TimeControl"/> object representing the parsed time settings.</returns>
    /// <exception cref="GGFParserException">Thrown when the time format is invalid or cannot be parsed.</exception>
    static TimeControl ParseTime(string timeStr)
    {
        var tokenizer = new Tokenizer { Input = timeStr };
        var times = new List<string>();
        string s;
        while ((s = tokenizer.ReadTo('/')) != string.Empty)
            times.Add(s);

        if (times.Count > 3)
            throw new GGFParserException("The representation of time was invalid. Valid format is \"[main_time]/[increment_time]/[extension_time]\".");

        var timesMs = new List<int>();
        for(var i = 0; i < times.Count; i++)
        {
            var clockTime = new List<string>();
            var clockTokenizer = new Tokenizer { Input = times[i] };
            while ((s = clockTokenizer.ReadTo(':')) != string.Empty)
            {
                // According to the GGF specification, some options can be specified after a comma,
                // but there are few reversi games that use them, so a comma is ignored.
                var idx = s.IndexOf(',');
                if(idx != -1)
                    s = s[..idx];
                clockTime.Add(s);
            }

            if (clockTime.Count > 3)
                throw new GGFParserException("The representation of clock time was invalid. Valid format is \"[hours]:[minutes]:[seconds]\".");

            var timeMs = 0;
            var unit = 1000;
            foreach(var t in clockTime.Reverse<string>())
            {
                if (!int.TryParse(t, out int v))
                    throw new GGFParserException("The value of hour, minute and second must be an integer.");

                timeMs += v * unit;
                unit *= 60;
            }
            timesMs.Add(timeMs);
        }

        var timeControl = new TimeControl();
        if (timesMs.Count > 0)
            timeControl.MainTimeMs = timesMs[0];

        if(timesMs.Count > 1)
            timeControl.IncrementMs = timesMs[1];

        return timeControl;
    }

    /// <summary>
    /// Parses a game result string from GGF format.
    /// </summary>
    /// <param name="resStr">The result string to parse.</param>
    /// <returns>A <see cref="GGFGameResult"/> object if parsing succeeds; otherwise, <c>null</c>.</returns>
    static GGFGameResult? ParseResult(string resStr)
    {
        var result = new GGFGameResult();
        var tokenizer = new Tokenizer { Input = resStr };
        tokenizer.ReadTo(':');

        if (!double.TryParse(tokenizer.Input, out double score))
            return null;

        result.FirstPlayerScore = score;
        switch (tokenizer.ReadNextChar())
        {
            case 'r':
                result.IsResigned = true;
                break;

            case 't':
                result.IsTimeout = true;
                break;

            case 's':
                result.IsMutual = true;
                break;
        }

        return result;
    }

    /// <summary>
    /// Parses a board position string from GGF format.
    /// </summary>
    /// <param name="posStr">The position string to parse.</param>
    /// <returns>A <see cref="Position"/> object representing the parsed board position.</returns>
    /// <exception cref="GGFParserException">Thrown when the position format is invalid or unsupported.</exception>
    static Position ParsePosition(string posStr)
    {
        var tokenizer = new Tokenizer { Input = posStr };
        if(tokenizer.ReadNextChar() != '8')
            throw new GGFParserException("Only 8x8 board is supported.");

        var pos = new Position(new Bitboard(0UL, 0UL), DiscColor.Black);
        var coord = BoardCoordinate.A1;
        char ch;
        while(coord <= BoardCoordinate.H8 && !tokenizer.IsEndOfString)
        {
            ch = tokenizer.ReadNextChar();
            switch (ch)
            {
                case '*':
                    pos.PutPlayerDiscAt(coord++);
                    break;

                case 'O':
                    pos.PutOpponentDiscAt(coord++);
                    break;

                case '-':
                    coord++;
                    break;

                default:
                    throw new GGFParserException($"Unexpected symbol \'{ch}\'");
            }
        }

        if (tokenizer.IsEndOfString)
            throw new GGFParserException("Missing side to move.");

        ch = tokenizer.ReadNextChar();
        if (ch == '*')
            pos.SideToMove = DiscColor.Black;
        else if (ch == 'O')
            pos.SideToMove = DiscColor.White;
        else
            throw new GGFParserException($"Unexpected symbol \'{ch}\'");

        return pos;
    }

    /// <summary>
    /// Parses a move string from GGF format.
    /// </summary>
    /// <param name="color">The color of the disc for this move.</param>
    /// <param name="moveStr">The move string to parse in format "coordinate[/eval_score[/time]]".</param>
    /// <returns>A <see cref="GGFMove"/> object representing the parsed move.</returns>
    /// <exception cref="GGFParserException">Thrown when the move format is invalid or cannot be parsed.</exception>
    static GGFMove ParseMove(DiscColor color, string moveStr)
    {
        var move = new GGFMove { Color = color };
        var moveInfo = new List<string>();
        var tokenizer = new Tokenizer { Input = moveStr };

        string s;
        while ((s = tokenizer.ReadTo('/')) != string.Empty)
            moveInfo.Add(s.ToLower());

        if (moveInfo.Count == 0)
            throw new GGFParserException("Coordinate was empty.");

        move.Coord = (moveInfo[0] == "pa") ? BoardCoordinate.Pass : Reversi.Utils.ParseCoordinate(moveInfo[0]);
        if (move.Coord == BoardCoordinate.Null)
            throw new GGFParserException($"Cannot parse \"{moveInfo[0]}\" as a coordinate.");

        if (moveInfo.Count > 1)
            if (double.TryParse(moveInfo[1], out double score))
                move.EvalScore = score;

        if (moveInfo.Count > 2)
            if (double.TryParse(moveInfo[2], out double time))
                move.Time = time;

        return move;
    }
}