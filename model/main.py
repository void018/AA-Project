import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load Data
df = pd.read_csv('data.csv')

# Explicitly drop the 'No' column to prevent data leakage
df = df.drop(columns=['No'], errors='ignore')

# 2. Define Features and Target
# Using exactly the 12 columns you specified
# features = ['V_A', 'V_B', 'V_C', 'I_A', 'I_B', 'I_C', 
#             'V_a', 'V_b', 'V_c', 'I_a', 'I_b', 'I_c']

# features = ['V_a', 'V_b', 'V_c']
features = ['V_A', 'V_B', 'V_C']
target = 'faulted'

X = df[features].values
y = df[target].values

# 3. Preprocessing
# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data - essential for V (high magnitude) vs I (lower magnitude)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# DataLoader for batching
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

# 4. Define the Model
class FaultDetector(nn.Module):
    def __init__(self):
        super(FaultDetector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() 
        )
        
    def forward(self, x):
        return self.net(x)

model = FaultDetector()

# 5. Loss and Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Training Loop
epochs = 50
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# 7. Final Evaluation
model.eval()
with torch.no_grad():
    y_pred_prob = model(X_test_t)
    y_pred = (y_pred_prob > 0.5).float()
    accuracy = (y_pred == y_test_t).float().mean()
    print(f"\nAccuracy after dropping 'No': {accuracy.item() * 100:.2f}%")