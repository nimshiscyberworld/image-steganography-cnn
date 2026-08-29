import torch


# ============================================================
# CONFIGURATION
# ============================================================

MESSAGE_LENGTH = 32          # maximum characters
BITS_PER_CHAR = 8
TOTAL_BITS = MESSAGE_LENGTH * BITS_PER_CHAR


# ============================================================
# TEXT → BITS
# ============================================================

def text_to_bits(text, max_length=MESSAGE_LENGTH):
    """
    Convert text into a fixed-length binary tensor.

    Example:
        "Hi"
        -> UTF-8 bytes
        -> binary
        -> tensor of 0s and 1s

    Returns:
        Tensor shape: [TOTAL_BITS]
    """

    # Convert text to UTF-8 bytes
    encoded = text.encode("utf-8")

    # Maximum number of bytes
    max_bytes = max_length

    # Check message length
    if len(encoded) > max_bytes:
        raise ValueError(
            f"Message is too long. Maximum is {max_bytes} UTF-8 bytes."
        )

    # Padding
    encoded = encoded.ljust(max_bytes, b"\x00")

    bits = []

    for byte in encoded:

        for bit_position in range(7, -1, -1):

            bit = (byte >> bit_position) & 1

            bits.append(bit)

    return torch.tensor(
        bits,
        dtype=torch.float32
    )


# ============================================================
# BITS → TEXT
# ============================================================

def bits_to_text(bits, max_length=MESSAGE_LENGTH):
    """
    Convert binary bits back into text.
    """

    # Convert tensor to list
    if isinstance(bits, torch.Tensor):
        bits = bits.detach().cpu().flatten().tolist()

    # Make sure values are binary
    bits = [
        1 if bit >= 0.5 else 0
        for bit in bits
    ]

    # Convert groups of 8 bits into bytes
    byte_values = []

    for i in range(0, len(bits), 8):

        byte = 0

        for bit in bits[i:i + 8]:

            byte = (byte << 1) | bit

        byte_values.append(byte)

    # Remove padding zeros
    byte_values = byte_values[:max_length]

    # Stop at first null byte
    if 0 in byte_values:

        byte_values = byte_values[:byte_values.index(0)]

    # Convert bytes → text
    try:

        text = bytes(byte_values).decode(
            "utf-8",
            errors="replace"
        )

    except Exception:

        text = ""

    return text


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_message = "Hello Bob"

    print("=" * 60)
    print("MESSAGE ENCODER TEST")
    print("=" * 60)

    print("Original text:")
    print(test_message)

    # Text → bits
    bits = text_to_bits(test_message)

    print("\nNumber of bits:")
    print(bits.numel())

    print("\nFirst 64 bits:")
    print(bits[:64])

    # Bits → text
    recovered_text = bits_to_text(bits)

    print("\nRecovered text:")
    print(recovered_text)

    print("\nTest successful:",
          test_message == recovered_text)

    print("=" * 60)