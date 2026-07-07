    import torch
    import torch.nn as nn

    # Tiny brain: 2 inputs, 16 neurons, 1 output
    class VerityBrain(nn.Module):
        def __init__(self):
            super().__init__
            self.net = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
        
        def forward(self, x):
            return self.net(x)

    # Training data: a + b = c
    data = [(1,2,3), (4,5,9), (10,20,30), (7,8,15)]
    X = torch.tensor([[a,b] for a,b,c in data], dtype=torch.float32)
    Y = torch.tensor([[c] for a,b,c in data], dtype=torch.float32)

    model = VerityBrain()
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    print("Training brain...")
    for epoch in range(100):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        opt.step()

    # Save the trained brain
    torch.save(model.state_dict(), "verity_brain.pt")
    print("Done! Saved as verity_brain.pt")
