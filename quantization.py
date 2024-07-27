import torch
from srgan_model import Generator

# Load the trained model
model = Generator()
state_dict = torch.load('Ohne_BN_200.pt', map_location='cpu')
model.load_state_dict(state_dict)

# Convert the weights to float16
model.half()  # Convert the model to half precision (float16)

# Save the quantized model
torch.save(model.state_dict(), 'srgan_ohneBN_float16.pt')
