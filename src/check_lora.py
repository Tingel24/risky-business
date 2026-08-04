from safetensors.torch import load_file

# Load the safetensors file
weights = load_file("../results/qwq_aligned_model/adapter_model.safetensors")
is_updated = False

for key, tensor in weights.items():
    if "lora_B" in key:
        # Check if the matrix is still all zeros
        if tensor.abs().sum().item() > 0:
            is_updated = True
            print(f"{key} successfully updated.")
            break

if not is_updated:
    print("All lora B matrices are zero. The model did not learn.")