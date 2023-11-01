import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from model_gru import *
from plotting import *

"""
python main_conv_diff.py
python main_conv_diff.py

"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # device

# Supposons que data soit un tableau numpy de forme (N, 4) où N est le nombre d'échantillons
# et chaque ligne est un quadruplet (t, x, y, u)
file_path = "solution_u_implicit_data.csv"
data = pd.read_csv(file_path)

def truncate_two_decimals(value):
    return int(value * 100) / 100.0

def truncate_three_decimals(value):
    return int(value * 1000) / 1000.0

u_at_0_003 = data[data["temps"] == 0.003]["u"].values

# Étape 2: Remplacez les valeurs u aux timesteps 0, 0.001 et 0.002 par cette valeur
for t in [0, 0.001, 0.002]:
    data.loc[data["temps"] == t, "u"] = u_at_0_003

data["x"] = data["x"].apply(truncate_two_decimals)
data["y"] = data["y"].apply(truncate_two_decimals)
data["temps"] = data["temps"].apply(truncate_three_decimals)

# Séparation des entrées et des cibles en utilisant .iloc
inputs = data.iloc[:, :-1].values  # Prend tout sauf la dernière colonne
targets = data.iloc[:, -1].values  # Prend seulement la dernière colonne



n = len(inputs)
idx_train = int(n * 0.4)
#idx_val = idx_train + int(n * 0.2)



# Fonction pour créer des séquences
def create_sequences(data, targets, sequence_length):
    X, Y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        Y.append(targets[i+sequence_length])
    return np.array(X), np.array(Y)

def create_sequences_with_full_intervals(data, targets, sequence_length, interval_length=1681):
    X, Y = [], []
    for i in range(0, len(data) - sequence_length, interval_length):
        for j in range(i, i + interval_length - sequence_length):
            X.append(data[j:j+sequence_length])
            Y.append(targets[j+sequence_length])
    return np.array(X), np.array(Y)

# Création de séquences pour chaque ensemble
# Création de séquences
sequence_length = 10



X, Y = create_sequences(inputs, targets, sequence_length)
#X_val, Y_val = create_sequences_with_full_intervals(X_val, Y_val, sequence_length)
#X_test, Y_test = create_sequences_with_full_intervals(X_test, Y_test, sequence_length)


X_train = X[:idx_train]
Y_train = Y[:idx_train]

X_test = X[idx_train:]
Y_test = Y[idx_train:]

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.5, random_state=42)


#print(X_test[-2])


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)


# Déplacer les tensors vers le GPU si disponible
if torch.cuda.is_available():
    X_train_tensor = X_train_tensor.cuda()
    Y_train_tensor = Y_train_tensor.cuda()
    X_val_tensor = X_val_tensor.cuda()
    Y_val_tensor = Y_val_tensor.cuda()
    X_test_tensor = X_test_tensor.cuda()
    Y_test_tensor = Y_test_tensor.cuda()

# Hyperparamètres
learning_rate = 0.001
epochs = 200
input_dim = 3  # (t, x, y)
hidden_dim = 128
layer_dim = 1 # CASE 1 (H =1) CASE 02 (H = 2)
output_dim = 1

# Initialisation du modèle, de la fonction de coût et de l'optimiseur
model = GRUModel(input_dim, hidden_dim, layer_dim, output_dim)

# TRANSFERED LEARNING
#model.load_state_dict(torch.load('model_path_case6.pth'))
#model.eval()  # Mettez le modèle en mode évaluation

if torch.cuda.is_available():
    model = model.cuda()
    
criterion = torch.nn.MSELoss()  # Erreur quadratique moyenne pour la régression
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


epoch = 0

val_outputs = model(X_val_tensor).squeeze(1)
val_loss = criterion(val_outputs, Y_val_tensor)

train_losses = []
val_losses = []
epochs_dat = []

# Boucle d'entraînement
while (epoch <= epochs and val_loss.item() > 2.5e-8):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_train_tensor).squeeze(1)
    loss = criterion(outputs, Y_train_tensor)

    # Backward pass et optimisation
    loss.backward()
    optimizer.step()

    # Évaluation sur l'ensemble de validation
    model.eval()
    val_outputs = model(X_val_tensor).squeeze(1)
    val_loss = criterion(val_outputs, Y_val_tensor)
    
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())
    epochs_dat.append(epoch)
    
    if epoch % 2 == 0:
        print(f"Epoch {epoch}/{epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")
        
    epoch += 1


""" #batch_size = 32  # ou toute autre taille qui convient à votre mémoire GPU
#num_batches = len(X_test) // batch_size

#model.eval()
#all_outputs = []

#for i in range(num_batches):
    #start_idx = i * batch_size
    #end_idx = start_idx + batch_size

    #X_batch = X_test_tensor[start_idx:end_idx]
    #if torch.cuda.is_available():
        #X_batch = X_batch.cuda()

    #with torch.no_grad():
        outputs = model(X_batch)
        all_outputs.append(outputs)

all_outputs = torch.cat(all_outputs, dim=0)

min_length = min(all_outputs.size(0), Y_test_tensor.size(0))
all_outputs = all_outputs[:min_length]

Y_test_tensor = Y_test_tensor[:min_length]

if torch.cuda.is_available():
    all_outputs = all_outputs.cuda()

loss = criterion(all_outputs.squeeze(1), Y_test_tensor)
print(f"Test Loss: {loss.item()}")


model.eval()  # Mettez le modèle en mode évaluation

print(X_test.shape)
print(Y_test.shape)
#print(Y_test)

"""
torch.save(model.state_dict(), 'model_gru_net_case1.pth')


# Supposons que 'data' est votre DataFrame et que 'temps' est la colonne contenant les temps
unique_times = data['temps'].unique()

# Supposons que X_test, Y_test et T_test soient vos données de test
# X_test contient les coordonnées (x, y), Y_test contient les valeurs réelles de u, et T_test contient les temps
# Supposons que 'data' est votre DataFrame
T_test = data[data['temps'].isin(unique_times)]['temps'].to_numpy()

max_errors = []

print("Results plotting")

for t in unique_times:
    if t >= 0.25 and t <= 0.599:  # Affichez les prédictions seulement après 0,2 secondes
        indices = np.where(X_test[:, -1, 0] == t)  # Utilisez le dernier pas de temps de chaque séquence
        #indices = np.where(X_test_2[:, 0] == t) 
        X_t = X_test[indices]
        Y_t = Y_test[indices]

        # Convertissez les données en tensors et déplacez-les sur le GPU si nécessaire
        X_tensor = torch.tensor(X_t, dtype=torch.float32)
        if torch.cuda.is_available():
            X_tensor = X_tensor.cuda()

        # Obtenez les prédictions du modèle
        with torch.no_grad():
            predicted_u = model(X_tensor).cpu().numpy().squeeze()

        # Calcul de l'erreur absolue
        error = np.abs(Y_t - predicted_u)
        max_errors.append(np.max(error))
        

        # Affichez les prédictions à l'aide des fonctions fournies
        plot_side_by_side(X_t[:, -1, 1], X_t[:, -1, 2], Y_t, predicted_u, error, t)
        #plot_side_by_side(X_t[:, 1], X_t[:, 2], Y_t, predicted_u, error, t)
        


average_max_error = np.mean(max_errors)
FINAL_max_error = np.max(max_errors)
print(f"Moyenne des erreurs maximales sur les pas de temps: {average_max_error}")
print(f"MAXIMUM des erreurs maximales sur les pas de temps: {FINAL_max_error}")

# 3. Afficher les pertes après l'entraînement
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses over Epochs')
plt.legend()
plt.grid(False)

# Mettre l'échelle en log sur l'axe des ordonnées
plt.yscale('log')

# 4. Enregistrer le graphique
plt.savefig('losses_plot6.png')
plt.show()