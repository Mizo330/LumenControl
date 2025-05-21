import pyaudio

NP_DTYPE_TO_PYAUDIO = {
    "float32": pyaudio.paFloat32,
    "int32":   pyaudio.paInt32,
    "int16":   pyaudio.paInt16,
    "int8":    pyaudio.paInt8,
    "uint8":   pyaudio.paUInt8,
}