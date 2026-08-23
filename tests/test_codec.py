import numpy as np

from quant_pipeline.codecs.uniform import dequantize, pack_unsigned, quantize, unpack_unsigned


def test_arbitrary_bit_pack_roundtrip():
    for bits in range(2, 9):
        values = np.arange(257, dtype=np.uint16) % (1 << bits)
        packed = pack_unsigned(values, bits)
        actual = unpack_unsigned(packed, len(values), bits)
        np.testing.assert_array_equal(actual, values)
        assert packed.nbytes == (len(values) * bits + 7) // 8


def test_quantize_roundtrip_and_stored_bytes():
    source = np.linspace(-2, 2, 513, dtype=np.float32).reshape(27, 19)
    encoded = quantize(source, bits=4, group_size=32)
    reconstructed = dequantize(encoded)
    assert reconstructed.shape == source.shape
    assert encoded.stored_bytes == encoded.packed.nbytes + encoded.scales.nbytes
    assert np.mean((source - reconstructed) ** 2) < 0.02

