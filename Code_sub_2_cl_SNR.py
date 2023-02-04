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

#
trial_time = 1000
#
#
SNR_base_db = -21
rbf_ratio = 0.08
#
#unit data input
data_train = genvec(1, 1000)
data_test = genvec(1, 250)


mu1 = [0]
sigma1 = [50]
sigma1 = np.diag(sigma1)
pro_cl1 = 0.55
data_train_cl1 = []
data_test_cl1 = []
tag_list_cl1 = []

mu2 = [500]
sigma2 = [75]
sigma2 = np.diag(sigma2)
pro_cl2 = 0.45
data_train_cl2 = []
data_test_cl2 = []
tag_list_cl2 = []


for i in range(len(data_train)):
    threshold = np.random.uniform(0, 1)
    if(threshold < pro_cl1):
        data_train_cl1 = np.append(data_train_cl1, gencl_row(data_train[i:i+1], mu1, sigma1))
    else:
        data_train_cl2 = np.append(data_train_cl2, gencl_row(data_train[i:i+1], mu2, sigma2))

for i in range(len(data_test)):
    threshold = np.random.uniform(0, 1)
    if (threshold < pro_cl1):
        data_test_cl1 = np.append(data_test_cl1, gencl_row(data_test[i:i + 1], mu1, sigma1))
    else:
        data_test_cl2 = np.append(data_test_cl2, gencl_row(data_test[i:i + 1], mu2, sigma2))

rbf_cl1 = np.linspace(mu1 - 2*sigma1, mu1 + 2*sigma1, num = int(rbf_ratio*len(data_train_cl1)))
rbf_cl2 = np.linspace(mu2 - 2*sigma2, mu2 + 2*sigma2, num = int(rbf_ratio*len(data_train_cl2)))

#rbf_cl1 = [-100, -50, 0, 50, 100]


a_cl1 = np.zeros((1, len(rbf_cl1))) + 1
a_cl2 = np.zeros((1, len(rbf_cl2))) + 1


delta_cl1 = 4*sigma1/(len(rbf_cl1)-1)
delta_cl2 = 4*sigma2/(len(rbf_cl2)-1)
print(delta_cl1)
print(delta_cl2)
gamma = 1

beta = 1.6e-3*50
alpha_cl1 = 0.35#beta*(1/delta_cl1)
alpha1 = 0
alpha_analysis_cl1 = gamma*alpha_cl1

beta = 1.6e-3*50
alpha_cl2 = 0.2#beta*(1/delta_cl2)
alpha1 = 10
alpha_analysis_cl2 = gamma*alpha_cl2


phi_syn_train_cl1_s = synth_phi_mat(data_train_cl1, rbf_cl1, alpha_cl1, alpha1)
phi_syn_train_cl2_s = synth_phi_mat(data_train_cl2, rbf_cl2, alpha_cl2, alpha1)
phi_syn_test_cl1_s = synth_phi_mat(data_test_cl1, rbf_cl1, alpha_cl1, alpha1)
phi_syn_test_cl2_s = synth_phi_mat(data_test_cl2, rbf_cl2, alpha_cl2, alpha1)

phi_train_cl1_a = synth_phi_mat(data_train_cl1, rbf_cl1, alpha_analysis_cl1, alpha1)
phi_train_cl2_a = synth_phi_mat(data_train_cl2, rbf_cl2, alpha_analysis_cl2, alpha1)
phi_test_cl1_a = synth_phi_mat(data_test_cl1, rbf_cl1, alpha_analysis_cl1, alpha1)
phi_test_cl2_a = synth_phi_mat(data_test_cl2, rbf_cl2, alpha_analysis_cl2, alpha1)

#noiseless signal
y_train_cl1 = get_y(phi_syn_train_cl1_s, a_cl1)
y_train_cl2 = get_y(phi_syn_train_cl2_s, a_cl2)
y_test_cl1 = get_y(phi_test_cl1_a, a_cl1)
y_test_cl2 = get_y(phi_test_cl2_a, a_cl2)

plt.plot(data_train_cl1, phi_train_cl1_a[:,0], 'o')
plt.plot(data_train_cl1, phi_train_cl1_a[:,10], 'o')
plt.plot(data_train_cl1, phi_train_cl1_a[:,20], 'o')
plt.plot(data_train_cl1, phi_train_cl1_a[:,30], 'o')
plt.ylabel('Response by a few basis')
plt.xlabel('1D input X, alpha = 1e-1')
plt.show()

y_syn = np.append(y_train_cl1, y_train_cl2)
avg_pow_y = np.mean(y_syn**2)
avg_pow_y_db = 10*np.log10(avg_pow_y)
mean_noise = 0

data_test_cl1_knnr = data_test_cl1.reshape(-1,1)
data_test_cl2_knnr = data_test_cl2.reshape(-1,1)

print(np.shape(data_test_cl1_knnr))
print(type(data_test_cl1_knnr))

MAE_list = []
MAE_list_knnr = []
neigh_cl1 = KNeighborsRegressor(n_neighbors=4)
neigh_cl2 = KNeighborsRegressor(n_neighbors=4)
data_test_cl1_knnr = data_test_cl1.reshape(-1, 1)
data_test_cl2_knnr = data_test_cl2.reshape(-1, 1)

for i in range(41):
    #-----------
    SNR_base_db = SNR_base_db + 1
    #-----------
    perform_list = []
    perform_list_knnr = []
    for j in range(1000):
        SNR_db = SNR_base_db
        noise_db = avg_pow_y_db - SNR_db
        noise_watts = 10 ** (noise_db / 10)

        noise = np.random.normal(mean_noise, np.sqrt(noise_watts), len(y_syn))
        noise_test_cl1 = np.random.normal(mean_noise, np.sqrt(noise_watts), len(y_test_cl1))
        noise_test_cl2 = np.random.normal(mean_noise, np.sqrt(noise_watts), len(y_test_cl2))

        noise_y = y_syn + noise
        noise_y_cl1_test = y_test_cl1 + noise_test_cl1
        noise_y_cl2_test = y_test_cl2 + noise_test_cl2



        noise_y_cl1 = noise_y[0:len(y_train_cl1)]
        noise_y_cl2 = noise_y[len(y_train_cl1):len(y_train_cl1)+len(y_train_cl2)]

        neigh_cl1.fit(data_test_cl1_knnr, noise_y_cl1_test)
        neigh_cl2.fit(data_test_cl2_knnr, noise_y_cl2_test)

        IM_cl1 = get_IM(phi_train_cl1_a, rbf_cl1)
        a_hat_cl1 = get_ahat(IM_cl1, phi_train_cl1_a, noise_y_cl1)
        IM_cl2 = get_IM(phi_train_cl2_a, rbf_cl2)
        a_hat_cl2 = get_ahat(IM_cl2, phi_train_cl2_a, noise_y_cl2)
        print(a_hat_cl1)

        y_pred_cl1 = get_y(phi_test_cl1_a, a_hat_cl1)
        y_pred_cl2 = get_y(phi_test_cl2_a, a_hat_cl2)

        diff_cl1 = np.abs(y_pred_cl1 - y_test_cl1)
        diff_cl2 = np.abs(y_pred_cl2 - y_test_cl2)

        diff = np.append(diff_cl1, diff_cl2)
        mean_diff = np.mean(diff)

        perform_list = np.append(perform_list, mean_diff)

        y_pred_cl1_knnr = neigh_cl1.predict(data_test_cl1_knnr)
        y_pred_cl2_knnr = neigh_cl2.predict(data_test_cl2_knnr)

        diff_knnr_cl1 = np.abs(y_pred_cl1_knnr-y_test_cl1)

        diff_knnr_cl2 = np.abs(y_pred_cl2_knnr-y_test_cl2)
        diff_knnr = np.append(diff_knnr_cl1, diff_knnr_cl2)

        mean_diff_knnr = np.mean(diff_knnr)

        perform_list_knnr = np.append(perform_list_knnr, mean_diff_knnr)

    MAE_per_cond = np.mean(perform_list)
    MAE_per_cond_knnr = np.mean(perform_list_knnr)
    MAE_list = np.append(MAE_list, MAE_per_cond)
    MAE_list_knnr = np.append(MAE_list_knnr, MAE_per_cond_knnr)
    if SNR_base_db == 20:
        print(a_hat_cl1)
        print(a_hat_cl2)

SNR_axis = np.linspace(-20, 20, 41)
print(SNR_axis)
print(MAE_list)
print(MAE_list_knnr)


plt.plot(SNR_axis, MAE_list, 'o')
plt.plot(SNR_axis, MAE_list_knnr, 'x')
plt.ylabel('MAE')
plt.xlabel('SNR(dB)')
plt.title('Performance of RBF-LSE and kNNR under different SNR in 1D space')
plt.show()

