import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline_cnn import (
    MessageEmbedder,
    Encoder,
    Decoder,
    MessageDecoder
)


# ============================================================
# D0 - DATA / LABEL / OPTIMIZER PIPELINE AUDIT
# ============================================================

print("=" * 70)
print("D0 - DATA / LABEL / OPTIMIZER PIPELINE AUDIT")
print("=" * 70)


# ------------------------------------------------------------
# 1. Device
# ------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nDevice:", device)


# ------------------------------------------------------------
# 2. Configuration
# ------------------------------------------------------------

BATCH_SIZE = 4
IMAGE_SIZE = 256
MESSAGE_BITS = 256


# ------------------------------------------------------------
# 3. Create models
# ------------------------------------------------------------

print("\n[1] Creating models...")

embedder = MessageEmbedder(
    message_bits=MESSAGE_BITS,
    feature_channels=64
).to(device)

encoder = Encoder().to(device)

decoder = Decoder().to(device)

message_decoder = MessageDecoder(
    message_bits=MESSAGE_BITS
).to(device)


# ------------------------------------------------------------
# 4. Create optimizer
# ------------------------------------------------------------

optimizer = torch.optim.Adam(
    list(embedder.parameters())
    + list(encoder.parameters())
    + list(decoder.parameters())
    + list(message_decoder.parameters()),
    lr=1e-4
)


# ------------------------------------------------------------
# 5. Check optimizer contains MessageEmbedder parameters
# ------------------------------------------------------------

print("\n[2] Checking optimizer parameters...")

embedder_parameter_ids = {
    id(parameter)
    for parameter in embedder.parameters()
}

optimizer_parameter_ids = {
    id(parameter)
    for group in optimizer.param_groups
    for parameter in group["params"]
}

missing_parameters = (
    embedder_parameter_ids - optimizer_parameter_ids
)

if len(missing_parameters) == 0:
    print("PASS: MessageEmbedder parameters are in optimizer.")
else:
    print("FAIL: Some MessageEmbedder parameters are missing!")


# ------------------------------------------------------------
# 6. Create random cover images
# ------------------------------------------------------------

print("\n[3] Creating random cover images...")

cover = torch.rand(
    BATCH_SIZE,
    1,
    IMAGE_SIZE,
    IMAGE_SIZE,
    device=device
)

print("Cover shape:", cover.shape)
print("Cover min:", cover.min().item())
print("Cover max:", cover.max().item())


# ------------------------------------------------------------
# 7. Generate random 256-bit messages
# ------------------------------------------------------------

print("\n[4] Generating random messages...")

message = torch.randint(
    0,
    2,
    (BATCH_SIZE, MESSAGE_BITS),
    device=device
).float()

print("Message shape:", message.shape)
print("Message dtype:", message.dtype)

print("Message unique values:",
      torch.unique(message).detach().cpu().tolist())

print("Message mean:",
      message.mean().item())


# ------------------------------------------------------------
# 8. Check that messages contain both 0 and 1
# ------------------------------------------------------------

print("\n[5] Checking message distribution...")

number_of_zeros = (message == 0).sum().item()
number_of_ones = (message == 1).sum().item()

print("Number of zeros:", number_of_zeros)
print("Number of ones :", number_of_ones)

if number_of_zeros > 0 and number_of_ones > 0:
    print("PASS: Message contains both 0 and 1.")
else:
    print("FAIL: Message contains only one value!")


# ------------------------------------------------------------
# 9. Test that new messages are actually different
# ------------------------------------------------------------

print("\n[6] Checking message randomness...")

message_2 = torch.randint(
    0,
    2,
    (BATCH_SIZE, MESSAGE_BITS),
    device=device
).float()

same_elements = (message == message_2).sum().item()

total_elements = message.numel()

similarity = same_elements / total_elements

print("Message 1 mean:", message.mean().item())
print("Message 2 mean:", message_2.mean().item())
print("Same-bit ratio:", similarity)

if similarity < 0.75:
    print("PASS: Messages appear random/different.")
else:
    print("WARNING: Messages may be too similar.")


# ------------------------------------------------------------
# 10. Forward pass
# ------------------------------------------------------------

print("\n[7] Testing forward pass...")

message_feature = embedder(message)

print("Message feature shape:",
      message_feature.shape)

stego = encoder(
    cover,
    message_feature
)

print("Stego shape:", stego.shape)

secret_feature = decoder(stego)

print("Decoder feature shape:",
      secret_feature.shape)

recovered = message_decoder(secret_feature)

print("Recovered message shape:",
      recovered.shape)


# ------------------------------------------------------------
# 11. Check recovered message range
# ------------------------------------------------------------

print("\n[8] Checking recovered message...")

print("Recovered min:",
      recovered.min().item())

print("Recovered max:",
      recovered.max().item())

print("Recovered mean:",
      recovered.mean().item())


# ------------------------------------------------------------
# 12. BCE loss test
# ------------------------------------------------------------

print("\n[9] Testing BCE loss...")

bce_loss = F.binary_cross_entropy(
    recovered,
    message
)

print("BCE loss:", bce_loss.item())

if torch.isfinite(bce_loss):
    print("PASS: BCE loss is valid.")
else:
    print("FAIL: BCE loss is NaN or infinite!")


# ------------------------------------------------------------
# 13. BCE sanity test
# ------------------------------------------------------------

print("\n[10] BCE sanity test...")

test_target = torch.tensor(
    [[0.0, 1.0, 0.0, 1.0]],
    device=device
)

test_prediction = torch.tensor(
    [[0.01, 0.99, 0.01, 0.99]],
    device=device
)

test_loss = F.binary_cross_entropy(
    test_prediction,
    test_target
)

print("Good prediction BCE:", test_loss.item())

if test_loss.item() < 0.1:
    print("PASS: BCE behaves correctly.")
else:
    print("WARNING: BCE result is unexpected.")


# ------------------------------------------------------------
# 14. Check gradient flow
# ------------------------------------------------------------

print("\n[11] Testing gradient flow...")

optimizer.zero_grad()

loss = F.binary_cross_entropy(
    recovered,
    message
)

loss.backward()


# ------------------------------------------------------------
# 15. Calculate gradient norm
# ------------------------------------------------------------

def gradient_norm(model):

    total = 0.0

    for parameter in model.parameters():

        if parameter.grad is not None:

            total += parameter.grad.detach().pow(2).sum().item()

    return total ** 0.5


embedder_grad = gradient_norm(embedder)
encoder_grad = gradient_norm(encoder)
decoder_grad = gradient_norm(decoder)
message_decoder_grad = gradient_norm(message_decoder)


print("\nGradient norms:")

print("MessageEmbedder :", embedder_grad)
print("Encoder        :", encoder_grad)
print("Decoder        :", decoder_grad)
print("MessageDecoder :", message_decoder_grad)


# ------------------------------------------------------------
# 16. Final D0 summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("D0 SUMMARY")
print("=" * 70)

print("Message shape          :", message.shape)
print("Message zeros          :", number_of_zeros)
print("Message ones           :", number_of_ones)
print("Message mean           :", message.mean().item())
print("Recovered mean         :", recovered.mean().item())
print("BCE loss               :", bce_loss.item())

print("\nGradient flow:")
print("MessageEmbedder        :", embedder_grad)
print("Encoder                :", encoder_grad)
print("Decoder                :", decoder_grad)
print("MessageDecoder         :", message_decoder_grad)

print("\n" + "=" * 70)
print("D0 COMPLETE")
print("=" * 70)