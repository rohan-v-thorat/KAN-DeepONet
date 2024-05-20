import numpy as np
import matplotlib.pyplot as plt

act = 'ReLU'
kan_model = 'G5D3W10'
mlp_model = 'D3W300'

kan_name = kan_model + act
mlp_name = mlp_model + act

kan_loss = np.load('result/KAN_loss'+kan_name+'.npy')
kan_test_loss = np.load('result/KAN_test_loss'+kan_name+'.npy')
mlp_loss = np.load('result/MLP_loss'+mlp_name+'.npy')
mlp_test_loss = np.load('result/MLP_test_loss'+mlp_name+'.npy')
epochs = np.linspace(1,10**4+1,10**4)

plt.plot(epochs[3000:],kan_loss[3000:])
plt.plot(epochs[3000:],kan_test_loss[3000:],'--')
plt.plot(epochs[3000:],mlp_loss[3000:])
plt.plot(epochs[3000:],mlp_test_loss[3000:],'--')
plt.legend(['KAN['+kan_model+'] train loss','KAN['+kan_model+'] test loss','MLP['+mlp_model+'] train loss','MLP['+mlp_model+'] test loss'])
plt.ylabel('MSE')
plt.xlabel('epochs')
plt.show()


# run for G3D2W10 SiLU again