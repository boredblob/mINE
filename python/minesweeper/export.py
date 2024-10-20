import torch
from torch import nn

game_width = 30
game_height = 16

class Network(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(game_width*game_height*3, 16),
      nn.ReLU(),
      nn.Linear(16, game_width*game_height*2)
    )

  def forward(self, t):
    t = self.net(t)
    return t
  
  def act(self, state):
    state_t = torch.as_tensor(state, dtype=torch.float32)
    q_values = self(state_t.unsqueeze(0))

    max_q = torch.argmax(q_values, dim=1)[0]
    return max_q.item()

model = Network()
model.load_state_dict(torch.load("C:/Users/omerk/Documents/GitHub/mINE/python/minesweeper/models/2024-10-2015-28-07", weights_only=True))
model.eval()

torch.onnx.dynamo_export(
model,
torch.randn(game_width*game_height*3),
"model.onnx")