import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

class WineQualityModel(nn.Module):
    def __init__(self):
        super(WineQualityModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 110),
            nn.ELU(),
            nn.Linear(110, 55),
            nn.ELU(),
            nn.Linear(55, 25),
            nn.ELU(),
            nn.Linear(25, 25),
            nn.ELU(),
            nn.Linear(25, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

df = pd.read_csv('pima-indians-diabetes.csv', header=None)

X = df.iloc[0:, :8].values.astype(np.float32)
y = df.iloc[0:, 8].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, stratify=y ,random_state=0)

X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train).reshape(-1, 1)
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test).reshape(-1, 1)

model = WineQualityModel()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

my_file = Path("model.pickle")
if not my_file.is_file():
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    torch.save(model, "model.pickle")
else:
    model = torch.load("model.pickle")

model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    test_loss = criterion(y_pred, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

y_pred_np = y_pred.numpy().flatten()
y_test_np = y_test_tensor.numpy().flatten()


accuracy = np.mean(np.abs(y_pred_np - y_test_np) <= 0.05)  
print(f'Accuracy: {accuracy:.4f}')

t_positive = 0
t_negative = 0
f_positive = 0
f_negative = 0

fansts = [ [0, 0, 0, 0], [0, 0, 0, 0] ]

for i in range(len(y_pred_np)):
    
    y_pred_scalar = y_pred_np[i]
    y_true_scalar = y_test_np[i]

    if y_pred_scalar < 0.5:
        y_pred_scalar = 0
    elif y_pred_scalar > 0.5:
        y_pred_scalar = 1

    if y_pred_scalar == y_true_scalar:
        fansts[int(y_pred_scalar)][0] = fansts[int(y_pred_scalar)][0] + 1
        for j in range(2):
            if((y_pred_scalar) == j):
                pass
            else:
                fansts[j][1] = fansts[j][1] + 1

    elif y_pred_scalar != y_true_scalar:
        fansts[int(y_true_scalar)][3] = fansts[int(y_true_scalar)][3] + 1
        fansts[int(y_pred_scalar)][2] = fansts[int(y_pred_scalar)][2] + 1
        for j in range(2):
            if((y_pred_scalar) == j or (y_true_scalar) == j):
                pass
            else:
                fansts[j][1] = fansts[j][1] + 1

print("Class \t Sensitivity \t Specificity \t Precision \t F1 Score")

for i in range(2):
    if fansts[i][0] + fansts[i][3] == 0:
        sensitivity = 0  
    else:
        sensitivity = fansts[i][0] / (fansts[i][0] + fansts[i][3]) #(True Positive)/(True Positive + False Negative) RECALL
    
    if fansts[i][1] + fansts[i][2] == 0:
        specificity = 0  
    else:
        specificity = fansts[i][1] / (fansts[i][1] + fansts[i][2]) #(True Negative)/(True Negative + False Positive)
    
    if fansts[i][0] + fansts[i][2] == 0:
        precision = 0  
    else:
        precision = fansts[i][0] / (fansts[i][0] + fansts[i][2])   #(True Positive)/(True Positive + False Positive)
   
    if precision + sensitivity == 0:
        f1_score = 0  
    else:
        f1_score = 2 * ((precision * sensitivity) / (precision + sensitivity)) #2x((precision x recall) / (precision + recall))

    print(str(i) + "\t" + str(round(sensitivity,2)) + "\t\t" + str(round(specificity, 2)) + "\t\t\t" + str(round(precision, 2)) + "\t" + str(round(f1_score, 2)))