namespace ACG2025_reference_implementation.Utils;

using System;
using System.Collections.Generic;
using System.Linq;

/// <summary>
/// A text tokenizer that can parse strings by splitting them on specified delimiter characters.
/// Provides functionality to read tokens, characters, and specific patterns from input text.
/// </summary>
/// <param name="delimiters">The collection of characters to use as delimiters when tokenizing.</param>
internal class Tokenizer(IEnumerable<char> delimiters)
{
    /// <summary>
    /// Gets the current position in the input string.
    /// </summary>
    public int Pos { get; private set; }
    
    /// <summary>
    /// Gets a value indicating whether the tokenizer has reached the end of the input string.
    /// </summary>
    public bool IsEndOfString => Pos == _input.Length;

    readonly char[] _delimiters = [.. delimiters];
    string _input = string.Empty;

    /// <summary>
    /// Initializes a new instance of the <see cref="Tokenizer"/> class with default delimiters (space, tab, carriage return, newline).
    /// </summary>
    public Tokenizer() : this([' ', '\t', '\r', '\n']) { }

    /// <summary>
    /// Gets or sets the input string to tokenize. Setting this property resets the position to 0.
    /// </summary>
    public string Input
    {
        get => _input;

        set
        {
            _input = value;
            Pos = 0;
        }
    }

    /// <summary>
    /// Reads the next character from the input, skipping any delimiters.
    /// </summary>
    /// <returns>The next non-delimiter character, or '\0' if the end of the string is reached.</returns>
    public char ReadNextChar()
    {
        SkipDelimiters();
        return (Pos == _input.Length) ? '\0' : _input[Pos++];
    }

    /// <summary>
    /// Reads the next token from the input. A token is a sequence of characters that are not delimiters.
    /// </summary>
    /// <returns>The next token as a string, or an empty string if the end of the input is reached.</returns>
    public string ReadNext()
    {
        SkipDelimiters();

        if (Pos == _input.Length)
            return string.Empty;

        var startPos = Pos;
        while (Pos < _input.Length && !_delimiters.Contains(_input[Pos]))
            Pos++;

        return _input[startPos..Pos];
    }

    /// <summary>
    /// Reads all characters from the current position up to (but not including) the specified end character.
    /// The position is advanced to just after the end character if found.
    /// </summary>
    /// <param name="end">The character to read up to.</param>
    /// <returns>The string from the current position to the end character, or to the end of input if the end character is not found.</returns>
    public string ReadTo(char end)
    {
        SkipDelimiters();

        if (Pos == _input.Length)
            return string.Empty;

        var span = _input.AsSpan(Pos);
        var idx = span.IndexOf(end);

        if (idx == -1)
        {
            Pos = _input.Length;
            return span[..span.Length].ToString();
        }

        Pos += idx + 1;
        return span[..idx].ToString();
    }

    /// <summary>
    /// Reads all remaining characters from the current position to the end of the input string.
    /// The position is advanced to the end of the input.
    /// </summary>
    /// <returns>The remaining characters in the input string, or an empty string if already at the end.</returns>
    public string ReadToEnd()
    {
        if (Pos == _input.Length)
            return string.Empty;

        var pos = Pos;
        Pos = _input.Length;
        return _input[pos..];
    }

    /// <summary>
    /// Advances the current position past any delimiter characters in the input string.
    /// </summary>
    void SkipDelimiters()
    {
        while (Pos < _input.Length && _delimiters.Contains(_input[Pos]))
            Pos++;
    }
}