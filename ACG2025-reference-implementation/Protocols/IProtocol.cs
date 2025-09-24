namespace ACG2025_reference_implementation.Protocols;

using ACG2025_reference_implementation.Engines;

internal interface IProtocol
{
    void Mainloop(Engine engine, string logPath);
    void Mainloop(Engine engine) => Mainloop(engine, string.Empty);
}