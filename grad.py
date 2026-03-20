import torch 
import torch.nn.functional as F

# Example of target with class indices
input = torch.randn(3, 5,requires_grad=True)  # Logits for 3 samples and 5 classes
target = torch.randint(5, (3,), dtype=torch.int64)
loss = F.cross_entropy(input, target)
print("Loss before backward:", loss.item())
loss.backward()

print("dLoss/dInput shape:", input.grad.shape)  # (3, 5)
print("dLoss/dInput values:\n", input.grad)


for i in range(input.shape[0]):
    prob = F.softmax(input[i], dim=0)
    print(f"Sample {i}: {prob}")

print("Input:", input)
print("Target:", target)
print("Loss:", loss.item())