import numpy as np
import matplotlib.pyplot as plt

## 
# act = 'ReLU'
# kan_model = 'G3D3W10'
# mlp_model = 'D3W30'

# kan_name = kan_model + act
# mlp_name = mlp_model + act

# kan_loss = np.load('result/KAN_loss'+kan_name+'.npy')
# kan_test_loss = np.load('result/KAN_test_loss'+kan_name+'.npy')
# mlp_loss = np.load('result/MLP_loss'+mlp_name+'.npy')
# mlp_test_loss = np.load('result/MLP_test_loss'+mlp_name+'.npy')
# epochs = np.linspace(1,10**4+1,10**4)

# plt.plot(epochs[3000:],kan_loss[3000:])
# plt.plot(epochs[3000:],kan_test_loss[3000:],'--')
# plt.plot(epochs[3000:],mlp_loss[3000:])
# plt.plot(epochs[3000:],mlp_test_loss[3000:],'--')
# plt.legend(['KAN['+kan_model+'] train loss','KAN['+kan_model+'] test loss','MLP['+mlp_model+'] train loss','MLP['+mlp_model+'] test loss'])
# plt.ylabel('MSE')
# plt.xlabel('epochs')
# plt.show()


## depth = 2
epochs = np.linspace(1,10**4+1,10**4)

kan_loss1 = np.load('result/KAN_lossG3D2W10ReLU.npy')
kan_test_loss1 = np.load('result/KAN_test_lossG3D2W10ReLU.npy')
kan_loss2 = np.load('result/KAN_lossG5D2W10ReLU.npy')
kan_test_loss2 = np.load('result/KAN_test_lossG5D2W10ReLU.npy')
kan_loss3 = np.load('result/KAN_lossG10D2W10ReLU.npy')
kan_test_loss3 = np.load('result/KAN_test_lossG10D2W10ReLU.npy')

mlp_loss1 = np.load('result/MLP_lossD2W30ReLU.npy')
mlp_test_loss1 = np.load('result/MLP_test_lossD2W30ReLU.npy')
mlp_loss2 = np.load('result/MLP_lossD2W100ReLU.npy')
mlp_test_loss2 = np.load('result/MLP_test_lossD2W100ReLU.npy')
mlp_loss3 = np.load('result/MLP_lossD2W300ReLU.npy')
mlp_test_loss3 = np.load('result/MLP_test_lossD2W300ReLU.npy')

plt.plot(epochs[3000:],kan_loss1[3000:])
plt.plot(epochs[3000:],kan_test_loss1[3000:],'--')
plt.plot(epochs[3000:],kan_loss2[3000:])
plt.plot(epochs[3000:],kan_test_loss2[3000:],'--')
plt.plot(epochs[3000:],kan_loss3[3000:])
plt.plot(epochs[3000:],kan_test_loss3[3000:],'--')

# plt.plot(epochs[3000:],mlp_loss1[3000:])
# plt.plot(epochs[3000:],mlp_test_loss1[3000:],'--')
# plt.plot(epochs[3000:],mlp_loss2[3000:])
# plt.plot(epochs[3000:],mlp_test_loss2[3000:],'--')
# plt.plot(epochs[3000:],mlp_loss3[3000:])
# plt.plot(epochs[3000:],mlp_test_loss3[3000:],'--')


# plt.legend(['KAN[G3D2W10ReLU] train loss','KAN[G3D2W10ReLU] test loss','MLP[D2W30ReLU] train loss','MLP[D2W30ReLU] test loss'])

# plt.legend(['KAN[G3D2W10ReLU] train loss','KAN[G3D2W10ReLU] test loss','KAN[G5D2W10ReLU] train loss','KAN[G5D2W10ReLU] test loss','KAN[G10D2W10ReLU] train loss','KAN[G10D2W10ReLU] test loss'])

# plt.legend(['MLP[D2W30ReLU] train loss','MLP[D2W30ReLU] test loss','MLP[D2W100ReLU] train loss','MLP[D2W100ReLU] test loss','MLP[D2W300ReLU] train loss','MLP[D2W300ReLU] test loss'])

# plt.legend(['KAN[G3D2W10ReLU] train loss','KAN[G3D2W10ReLU] test loss','KAN[G5D2W10ReLU] train loss','KAN[G5D2W10ReLU] test loss','KAN[G10D2W10ReLU] train loss','KAN[G10D2W10ReLU] test loss',\
#             'MLP[D2W30ReLU] train loss','MLP[D2W30ReLU] test loss','MLP[D2W100ReLU] train loss','MLP[D2W100ReLU] test loss','MLP[D2W300ReLU] train loss','MLP[D2W300ReLU] test loss'])
plt.ylabel('MSE')
plt.xlabel('epochs')
plt.ylim([0,0.001])
plt.show()

## depth = 3
# epochs = np.linspace(1,10**4+1,10**4)

# kan_loss1 = np.load('result/KAN_lossG3D3W10ReLU.npy')
# kan_test_loss1 = np.load('result/KAN_test_lossG3D3W10ReLU.npy')
# kan_loss2 = np.load('result/KAN_lossG5D3W10ReLU.npy')
# kan_test_loss2 = np.load('result/KAN_test_lossG5D3W10ReLU.npy')
# kan_loss3 = np.load('result/KAN_lossG10D3W10ReLU.npy')
# kan_test_loss3 = np.load('result/KAN_test_lossG10D3W10ReLU.npy')

# mlp_loss1 = np.load('result/MLP_lossD3W30ReLU.npy')
# mlp_test_loss1 = np.load('result/MLP_test_lossD3W30ReLU.npy')
# mlp_loss2 = np.load('result/MLP_lossD3W100ReLU.npy')
# mlp_test_loss2 = np.load('result/MLP_test_lossD3W100ReLU.npy')
# mlp_loss3 = np.load('result/MLP_lossD3W300ReLU.npy')
# mlp_test_loss3 = np.load('result/MLP_test_lossD3W300ReLU.npy')

# plt.plot(epochs[3000:],kan_loss1[3000:])
# plt.plot(epochs[3000:],kan_test_loss1[3000:],'--')
# # plt.plot(epochs[3000:],kan_loss2[3000:])
# # plt.plot(epochs[3000:],kan_test_loss2[3000:],'--')
# # plt.plot(epochs[3000:],kan_loss3[3000:])
# # plt.plot(epochs[3000:],kan_test_loss3[3000:],'--')

# plt.plot(epochs[3000:],mlp_loss1[3000:])
# plt.plot(epochs[3000:],mlp_test_loss1[3000:],'--')
# # plt.plot(epochs[3000:],mlp_loss2[3000:])
# # plt.plot(epochs[3000:],mlp_test_loss2[3000:],'--')
# # plt.plot(epochs[3000:],mlp_loss3[3000:])
# # plt.plot(epochs[3000:],mlp_test_loss3[3000:],'--')


# plt.legend(['KAN[G3D3W10ReLU] train loss','KAN[G3D3W10ReLU] test loss','MLP[D3W30ReLU] train loss','MLP[D3W30ReLU] test loss'])

# # plt.legend(['KAN[G3D3W10ReLU] train loss','KAN[G3D3W10ReLU] test loss','KAN[G5D3W10ReLU] train loss','KAN[G5D3W10ReLU] test loss','KAN[G10D3W10ReLU] train loss','KAN[G10D3W10ReLU] test loss'])

# # plt.legend(['MLP[D3W30ReLU] train loss','MLP[D3W30ReLU] test loss','MLP[D3W100ReLU] train loss','MLP[D3W100ReLU] test loss','MLP[D3W300ReLU] train loss','MLP[D3W300ReLU] test loss'])

# # plt.legend(['KAN[G3D3W10ReLU] train loss','KAN[G3D3W10ReLU] test loss','KAN[G5D3W10ReLU] train loss','KAN[G5D3W10ReLU] test loss','KAN[G10D3W10ReLU] train loss','KAN[G10D3W10ReLU] test loss',\
# #             'MLP[D3W30ReLU] train loss','MLP[D3W30ReLU] test loss','MLP[D3W100ReLU] train loss','MLP[D3W100ReLU] test loss','MLP[D3W300ReLU] train loss','MLP[D3W300ReLU] test loss'])
# plt.ylabel('MSE')
# plt.xlabel('epochs')
# # plt.ylim([0,0.001])
# plt.show()