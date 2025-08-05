import torch
from models.image_model import ImageModerationModel

model = ImageModerationModel()
model.load_state_dict(torch.load("pretrained/efficientnet_multiclass_trained.pth"))
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.jit.script(model).save("converted_efficientnet.pt")