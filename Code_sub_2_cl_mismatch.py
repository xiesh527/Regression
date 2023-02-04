from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import Rbf
np.set_printoptions(suppress=False)
from sklearn.neighbors import KNeighborsRegressor

def genvec(dim,num):
    data = np.random.randn(num, dim)
    return data

def genmu(val, vec):
    mu = [val for i in range(len(vec))]
    return mu

def gencl_row(cl, mu, sigmas):
    result = np.dot(cl, sigmas) + mu
    return result


def synth_phi_mat(data, rbf, alpha, alpha1):
    phi = np.zeros((1, len(rbf)))
    for i in range(len(data)):
        data_row = data[i]
        phi_buf = []
        for j in range(len(rbf)):
            per_phi = np.exp(-(alpha * np.abs(data_row - rbf[j])) ** 2) * np.exp(np.exp(-(alpha1 * np.abs(data_row - rbf[j]))) - 1)
            phi_buf = np.append(phi_buf, per_phi)
        phi = np.vstack([phi, phi_buf])
    phi = phi[1:len(phi)]
    return(phi)


def get_ahat(IM, phi, y):
    inter = np.dot(np.linalg.inv(IM), phi.T)
    a_hat = np.dot(inter, y)
    return a_hat

def get_y(phi, a):
    y = []
    for i in range(len(phi)):
        signal_buf = np.dot(phi[i], a.T)
        y = np.append(y, signal_buf)
    return y

def get_IM(phi, rbf):
    IM = np.zeros((len(rbf), len(rbf)))
    for i in range(len(phi)):
        per_IM = np.outer(phi[i].T, phi[i])
        IM = IM + per_IM
    return IM

def get_vnaa(EIM, psi):
    vnaa = EIM[:, 1:len(psi)]
    vnaa = vnaa[1:len(psi.T)]
    return vnaa

def get_vna1(EIM):
    vna1 = EIM[:, 0]
    vna1 = vna1[1:len(vna1)]
    return vna1

#-------
trial_time = 1000
#-------
#-------
SNR_db = 20
rbf_ratio = 0.08
gamma_base = np.linspace(0.1, 2, 40)
#-------

data_train = genvec(1, 1000)
data_test = genvec(1, 250)

mu1 = [0]
sigma1 = [50]
sigma1 = np.diag(sigma1)
pro_cl1 = 0.55
data_train_cl1 = []
data_test_cl1 = []
tag_list_cl1 = []

mu2 = [300]
sigma2 = [75]
sigma2 = np.diag(sigma2)
pro_cl2 = 0.45
data_train_cl2 = []
data_test_cl2 = []
tag_list_cl2 = []

data_train_syn =[]
for i in range(len(data_train)):
    threshold = np.random.uniform(0, 1)
    if(threshold < pro_cl1):
        data_train_cl1 = np.append(data_train_cl1, gencl_row(data_train[i:i+1], mu1, sigma1))
    else:
        data_train_cl2 = np.append(data_train_cl2, gencl_row(data_train[i:i+1], mu2, sigma2))

data_train_global = np.append(data_train_cl1, data_train_cl2)

for i in range(len(data_test)):
    threshold = np.random.uniform(0, 1)
    if (threshold < pro_cl1):
        data_test_cl1 = np.append(data_test_cl1, gencl_row(data_test[i:i + 1], mu1, sigma1))
    else:
        data_test_cl2 = np.append(data_test_cl2, gencl_row(data_test[i:i + 1], mu2, sigma2))

data_test_global = np.append(data_test_cl1, data_test_cl2)
rbf_cl1 = np.linspace(mu1 - 2*sigma1, mu1 + 2*sigma1, num = int(rbf_ratio*len(data_train_cl1)))
rbf_cl2 = np.linspace(mu2 - 2*sigma2, mu2 + 2*sigma2, num = int(rbf_ratio*len(data_train_cl2)))

rbf_global = np.linspace(mu1 - 2*sigma1, mu2 + 2*sigma2, num = int(rbf_ratio*len(data_train)))
print(rbf_global)

a_cl1 = np.zeros((1, len(rbf_cl1))) + 1
a_cl2 = np.zeros((1, len(rbf_cl2))) + 1

delta_cl1 = 4*sigma1/(len(rbf_cl1)-1)
delta_cl2 = 4*sigma2/(len(rbf_cl2)-1)

print(delta_cl1)
print(delta_cl2)

beta = 1.6e-3*50
alpha_cl1 = 0.2#0.35#beta*(1/delta_cl1)
alpha1 = 10
alpha_analysis_cl1 = gamma_base*alpha_cl1

beta = 1.6e-3*50
alpha_cl2 = 0.2
alpha1 = 10
alpha_analysis_cl2 = gamma_base*alpha_cl2

alpha_global = alpha_cl1*pro_cl1 + alpha_cl2*pro_cl2

phi_syn_train_cl1_s = synth_phi_mat(data_train_cl1, rbf_cl1, alpha_cl1, alpha1)
phi_syn_train_cl2_s = synth_phi_mat(data_train_cl2, rbf_cl2, alpha_cl2, alpha1)


phi_test_cl1_a = synth_phi_mat(data_test_cl1, rbf_cl1, alpha_cl1, alpha1)
phi_test_cl2_a = synth_phi_mat(data_test_cl2, rbf_cl2, alpha_cl2, alpha1)

y_train_cl1 = get_y(phi_syn_train_cl1_s, a_cl1)
y_train_cl2 = get_y(phi_syn_train_cl2_s, a_cl2)
y_test_cl1 = get_y(phi_test_cl1_a, a_cl1)
y_test_cl2 = get_y(phi_test_cl2_a, a_cl2)

y_test = np.append(y_test_cl1, y_test_cl2)
y_syn = np.append(y_train_cl1, y_train_cl2)

y_syn_test = np.append(y_test_cl1, y_test_cl2)
avg_pow_y = np.mean(y_syn**2)
avg_pow_y_db = 10*np.log10(avg_pow_y)
mean_noise = 0

SNR_db = 0
avg_pow_y = np.mean(y_syn ** 2)
noise_db = avg_pow_y_db - SNR_db
noise_watts = 10 ** (noise_db / 10)

MAE_list = []
MAE_list_knnr = []
neigh_cl1 = KNeighborsRegressor(n_neighbors=4)
neigh_cl2 = KNeighborsRegressor(n_neighbors=4)
neigh_global  = KNeighborsRegressor(n_neighbors=4)
data_test_cl1_knnr = data_test_cl1.reshape(-1, 1)
data_test_cl2_knnr = data_test_cl2.reshape(-1, 1)
data_test_global_knnr = data_test_global.reshape(-1,1)

MAE_all_i = []
MAE_all_i_knnr = []
MAE_all_i_global = []
MAE_all_i_knnr_global = []
print(np.shape(gamma_base))
for i in range(len(gamma_base)):
    gamma_buf = gamma_base[i]
    phi_a_cl1_ana = synth_phi_mat(data_train_cl1, rbf_cl1, gamma_buf*alpha_cl1, alpha1)
    phi_a_cl2_ana = synth_phi_mat(data_train_cl2, rbf_cl2, gamma_buf*alpha_cl2, alpha1)
    phi_a_cl1_test = synth_phi_mat(data_test_cl1, rbf_cl1, gamma_buf*alpha_cl1, alpha1)
    phi_a_cl2_test = synth_phi_mat(data_test_cl2, rbf_cl2, gamma_buf*alpha_cl2, alpha1)
    phi_global_ana = synth_phi_mat(data_train_global, rbf_global, gamma_buf*alpha_global, alpha1)
    phi_global_ana_test = synth_phi_mat(data_test_global, rbf_global, gamma_buf*alpha_global, alpha1)

    IM_cl1 = get_IM(phi_a_cl1_ana, rbf_cl1)
    IM_cl2 = get_IM(phi_a_cl2_ana, rbf_cl2)
    IM_global = get_IM(phi_global_ana, rbf_global)

    MAE_all_j = []
    MAE_all_j_knnr = []
    MAE_all_j_global = []
    MAE_all_j_knnr_global = []
    for j in range(1000):
        print('i : j =', i, ':', j)
        noise = np.random.normal(mean_noise, np.sqrt(noise_watts), len(y_syn))
        noise_test = np.random.normal(mean_noise, np.sqrt(noise_watts), len(y_syn_test))

        noise = np.random.normal(mean_noise, np.sqrt(noise_watts), len(y_syn))

        noise_y = y_syn + noise
        noise_y_cl1_s = noise_y[0:len(y_train_cl1)]
        noise_y_cl2_s = noise_y[len(y_train_cl1):len(y_train_cl1) + len(y_train_cl2)]

        plt.plot(data_train_cl1, noise_y_cl1_s, 'o')
        plt.ylabel('Output signal')
        plt.xlabel('Input axis')
        plt.title("The output signal at SNR = 0")
        plt.show()

        a_hat_cl1 = get_ahat(IM_cl1, phi_a_cl1_ana, noise_y_cl1_s)
        a_hat_cl2 = get_ahat(IM_cl2, phi_a_cl2_ana, noise_y_cl2_s)
        a_hat_global = get_ahat(IM_global, phi_global_ana, noise_y)

        y_pred_cl1 = get_y(phi_a_cl1_test, a_hat_cl1)
        y_pred_cl2 = get_y(phi_a_cl2_test, a_hat_cl2)
        y_pred_global = get_y(phi_global_ana_test, a_hat_global)

        y_pred = np.append(y_pred_cl1, y_pred_cl2)

        diff_per_j = np.abs(y_pred-y_test)
        diff_per_j_global = np.abs(y_pred_global-y_test)
        MAE_per_j = np.mean(diff_per_j)
        MAE_per_j_global = np.mean(diff_per_j_global)
        MAE_all_j = np.append(MAE_all_j, MAE_per_j)
        MAE_all_j_global = np.append(MAE_all_j_global, MAE_per_j_global)

        noise_test = np.random.normal(mean_noise, np.sqrt(noise_watts), len(y_test))
        noise_y_test = y_test + noise_test
        noise_y_cl1_test = noise_y[0:len(y_test_cl1)]
        noise_y_cl2_test = noise_y[len(y_test_cl1):len(y_test_cl1) + len(y_test_cl2)]

        neigh_cl1.fit(data_test_cl1_knnr, noise_y_cl1_test)
        neigh_cl2.fit(data_test_cl2_knnr, noise_y_cl2_test)
        neigh_global.fit(data_test_global_knnr, noise_y_test)

        y_pred_cl1_knnr = neigh_cl1.predict(data_test_cl1_knnr)
        y_pred_cl2_knnr = neigh_cl2.predict(data_test_cl2_knnr)
        y_pred_knnr = np.append(y_pred_cl1_knnr, y_pred_cl2_knnr)

        y_pred_global_knnr = neigh_global.predict(data_test_global_knnr)

        diff_per_j_knnr_cl1 = np.abs(y_pred_cl1_knnr-y_test_cl1)
        diff_per_j_knnr_cl2 = np.abs(y_pred_cl2_knnr-y_test_cl2)

        diff_per_j_knnr = np.append(diff_per_j_knnr_cl1,diff_per_j_knnr_cl2)
        diff_per_j_knnr_global = np.abs(y_pred_global_knnr-y_test)
        MAE_per_j_knnr= np.mean(diff_per_j_knnr)
        MAE_per_j_knnr_global = np.mean(diff_per_j_knnr_global)
        MAE_all_j_knnr = np.append(MAE_all_j_knnr, MAE_per_j_knnr)
        MAE_all_j_knnr_global = np.append(MAE_all_j_knnr_global, MAE_per_j_knnr_global)

    MAE_per_i = np.mean(MAE_all_j)
    MAE_per_i_global = np.mean(MAE_all_j_global)
    MAE_per_i_knnr_global = np.mean(MAE_all_j_knnr_global)
    MAE_all_i_knnr_global = np.append(MAE_all_i_knnr_global, MAE_per_i_knnr_global)
    MAE_all_i = np.append(MAE_all_i, MAE_per_i)
    MAE_all_i_global = np.append(MAE_all_i_global, MAE_per_i_global)
    #print(MAE_all_i)
    MAE_per_i_knnr = np.mean(MAE_all_j_knnr)
    MAE_all_i_knnr = np.append(MAE_all_i_knnr, MAE_per_i_knnr)

    #print('knnr MAE', MAE_all_i_knnr)




print(MAE_all_i)
print(MAE_all_i_knnr)
xa = np.linspace(0, 40, 40)
plt.plot(xa, MAE_all_i_knnr, 'o')
plt.plot(xa, MAE_all_i_knnr_global, 'x')
plt.ylabel('MAE')
plt.xlabel('Trial count')
plt.title('Performance of global and localized kNNR')
plt.show()

plt.plot(gamma_base, MAE_all_i, 'o')
plt.plot(gamma_base, MAE_all_i_global, 'x')
plt.ylabel('MAE')
plt.xlabel('Mismatch alpha_a/alpha_s')
plt.title('Performance of global and localized RBF-LSE')
plt.show()






