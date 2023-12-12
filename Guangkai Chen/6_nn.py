import json
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


####################################################
def load_json_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_data_to_json(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def load_csv_to_matrix(filename): # Never used
    df = pd.read_csv(filename)
    return df.iloc[:, 1:].values.tolist()
####################################################


import torch.nn as nn
import torch.nn.functional as F

class RegressionNN(nn.Module):
    def __init__(self, input_size):
        super(RegressionNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)  # Input layer
        self.fc2 = nn.Linear(32, 32)  # Hidden layer
        #self.fc3 = nn.Linear(64, 64)  # Hidden layer
        #self.fc4 = nn.Linear(64, 64)  # Hidden layer
        self.fc5 = nn.Linear(32, 1)   # Output layer

        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu') # He initialization
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # Define forward pass through the network using ReLU
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        #x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        x = self.fc5(x)  # No activation function for the output layer in regression
        return x
####################################################


from torch.utils.data import Dataset, DataLoader
import torch

#def write_to_file(file_path, data):
        #with open(file_path, 'a') as file:
            #file.write(data + '\n')

class CustomDataloader(Dataset):
    

    def __init__(self, avg, pearson, spearman, score, revenue):
        #self.matrix = matrix
        #self.scores = scores
        #self.correlations = correlations  # This should include both Pearson coefficients and p-values
        self.avg = avg
        self.pearson = pearson
        self.spearman = spearman
        self.revenue = revenue
        self.score = score
        #self.rank = rank

    def __len__(self):
        return len(self.avg) #3

    def __getitem__(self, idx):
        
        # Constructing the full feature vector
        features = [self.avg[idx], self.pearson[idx], self.spearman[idx], self.score[idx]]#[self.avg[idx] + [self.scores[idx]] + correlation_features] , self.score[idx], self.rank[idx]
        target = self.revenue[idx]

        #print("features: ", features)
        #print("target: ", target)

        #features_str = ', '.join(map(str, features))
        #target_str = str(target)
        #data_str = f"Features: {features_str}, Target: {target_str}"
        
        # Write to file
        #write_to_file('dataloader_output.txt', data_str)

        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
####################################################


def main():
    
    # Parameters
    input_size = 4 #4928
    batch_size = 200
    alpha = 0.001
    num_epochs = 1000

    # Load data
    #correlation = load_json_data('category_correlations.json')
    #matrix = load_csv_to_matrix('hu_distance_matrix_scaled.csv')
    hotels = load_json_data('hu_hotel_dropped.json') #load_json_data('hu_hotel_weighted.json') load_json_data('hu_hotel_filtered.json')
    score = [hotel.get('Total Score', 0) for hotel in hotels] # for hotel in hotels
    rank = [hotel.get('Rank', 0) for hotel in hotels]
    revenue = [hotel.get('revenue', 0) for hotel in hotels]
    avg = [hotel.get('avg_distance', 0) for hotel in hotels]
    pearson = [hotel.get('pearson_weighted_avg_distance', 0) for hotel in hotels]
    spearman = [hotel.get('spearman_weighted_avg_distance', 0) for hotel in hotels]

####################################################
    train_score = [hotel.get('Total Score', 0) for hotel in hotels][:-50]
    train_revenue = [hotel.get('revenue', 0) for hotel in hotels][:-50]
    train_avg = [hotel.get('avg_distance', 0) for hotel in hotels][:-50]
    train_pearson = [hotel.get('pearson_weighted_avg_distance', 0) for hotel in hotels][:-50]
    train_spearman = [hotel.get('spearman_weighted_avg_distance', 0) for hotel in hotels][:-50]

    features = np.array([train_avg, train_pearson, train_spearman, train_score]).T
    target = np.array(train_revenue).reshape(-1, 1)

    # Initialize the StandardScaler
    scaler_features = MinMaxScaler() #StandardScaler()
    scaler_target = MinMaxScaler() #StandardScaler()

    scaler_features.fit(features)
    scaler_target.fit(target)

    # Normalize the features
    normalized_features = scaler_features.fit_transform(features)
    normalized_target = scaler_target.transform(target)

    # Splitting normalized features back into individual lists
    train_avg, train_pearson, train_spearman, train_score = normalized_features.T.tolist()
    train_revenue = normalized_target.flatten().tolist()
    #print("train_avg", train_avg)
####################################################

    dataset = CustomDataloader(train_avg, train_pearson, train_spearman, train_score, train_revenue)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = RegressionNN(input_size)

    # Loss and optimizer
    criterion = nn.MSELoss() #L1Loss() #
    optimizer = optim.Adam(model.parameters(), lr=alpha)

    #print(f"DataLoader size: {len(data_loader)}")
    if len(data_loader) == 0:
        print("DataLoader is empty. Exiting.")
        return

    # Training loop
    losses = []

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, (features, targets) in enumerate(data_loader):
            targets = targets.view(-1, 1)  # Ensure correct target shape

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient Clipping
            optimizer.step()

            epoch_loss += loss.item()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # Average loss for the epoch
        #epoch_loss /= len(data_loader)
        losses.append(epoch_loss)

    print("Avg training loss: ", sum(losses)/num_epochs)
    # Save the model
    torch.save(model.state_dict(), 'model.pth')

    plt.plot(losses, linestyle='dotted')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over epochs')
    plt.show()

    ##################################################### TESTING
    test_score = [hotel.get('Total Score', 0) for hotel in hotels][-50:]
    test_revenue = [hotel.get('revenue', 0) for hotel in hotels][-50:]
    test_avg = [hotel.get('avg_distance', 0) for hotel in hotels][-50:]
    test_pearson = [hotel.get('pearson_weighted_avg_distance', 0) for hotel in hotels][-50:]
    test_spearman = [hotel.get('spearman_weighted_avg_distance', 0) for hotel in hotels][-50:]

    test_features = np.array([test_avg, test_pearson, test_spearman, test_score]).T
    test_target = np.array(test_revenue).reshape(-1, 1)

    normalized_test_features = scaler_features.transform(test_features)
    normalized_test_target = scaler_target.transform(test_target)

    # Splitting normalized features back into individual lists
    test_epochs = 500
    test_batch = 50

    test_avg, test_pearson, test_spearman, test_score = normalized_test_features.T.tolist()
    test_revenue = normalized_test_target.flatten().tolist()

    test_dataset = CustomDataloader(test_avg, test_pearson, test_spearman, test_score, test_revenue)
    test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=True)

    epoch_losses = []
    model.eval()
    
    for epoch in range(test_epochs):
        epoch_loss = 0
        for i, (features, targets) in enumerate(test_loader):
            #print("targets: ", targets)
            targets = targets.view(-1, 1)
            outputs = model(features)
            #print("outputs: ", outputs)
            loss = criterion(outputs, targets)
            epoch_loss += loss.item()

        epoch_losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{test_epochs}], Loss: {epoch_loss:.20f}')
    
    print("Avg testing loss: ", sum(epoch_losses)/test_epochs)

    plt.plot(epoch_losses, linestyle='dotted')
    plt.xlabel('Epoch')
    plt.ylabel('Losses')
    plt.ylim(0.001, 0.005)
    plt.title('Testing Loss over epochs')
    plt.show()


if __name__ == "__main__":
    main()