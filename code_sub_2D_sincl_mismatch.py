from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import Rbf
np.set_printoptions(suppress=False)
from sklearn.neighbors import KNeighborsRegressor
from scipy.spatial import distance

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
            per_phi = np.exp(-(alpha * np.linalg.norm(data_row - rbf[j])) ** 2) * np.exp(np.exp(-(alpha1 * np.linalg.norm(data_row - rbf[j]))) - 1)
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
def get_ahat_at_SNR(p_avg_signal_db, SNR_db, signal, IM, phi):
    noise_db = p_avg_signal_db - SNR_db
    noise_watts = 10 ** (noise_db / 10)
    noise = np.random.normal(0, np.sqrt(noise_watts), len(signal))

    noise_y = signal + noise
    a_hat = get_ahat(IM, phi, noise_y)
    return a_hat
def get_MAE_per_mcs(y_test, phi_test, a_hat):
    y_pred = get_y(phi_test, a_hat)
    diff_mean = np.mean(np.abs(y_pred - y_test))
    return diff_mean

def get_noisy_y(p_avg_signal_db, SNR_db, signal):
    noise_db = p_avg_signal_db - SNR_db
    noise_watts = 10 ** (noise_db / 10)
    noise = np.random.normal(0, np.sqrt(noise_watts), len(signal))
    noise_y = signal + noise
    return noise_y

def get_vnaa(EIM, psi):
    vnaa = EIM[:, 1:len(psi)]
    vnaa = vnaa[1:len(psi.T)]
    return vnaa

def get_vna1(EIM):
    vna1 = EIM[:, 0]
    vna1 = vna1[1:len(vna1)]
    return vna1


rbf_ratio = 0.08
data_train = genvec(2,1000)
data_test = genvec(2, 250)

plt.plot(data_train[:,0], data_train[:,1], 'o')
plt.show()

gamma = np.linspace(0.1, 2, 65)

mu1 = [0, 0]
sigma1 = [50, 50]
sigma1_x1 = sigma1[0]
sigma1_x2 = sigma1[1]

sigma1 = np.diag(sigma1)


data_train_cl1 = [0, 0]
for i in range(len(data_train)):
    data_train_cl1 = np.vstack([data_train_cl1, gencl_row(data_train[i:i+1], mu1, sigma1)])
data_train_cl1 = data_train_cl1[1:len(data_train_cl1)]
plt.plot(data_train_cl1[:,0], data_train_cl1[:,1], 'o')
plt.title('Input domain with single Gaussian Component')
plt.show()
data_test_cl1 = [0, 0]
for i in range(len(data_test)):
    data_test_cl1 = np.vstack([data_test_cl1, gencl_row(data_test[i:i+1], mu1, sigma1)])
data_test_cl1 = data_test_cl1[1:len(data_test_cl1)]
print(np.shape(data_test_cl1))

num_rbf = rbf_ratio*len(data_train_cl1)
num_rbf_axis = int(np.round(np.sqrt(num_rbf)))
rbf_cl1_x1 = np.linspace(mu1[0] - 2*sigma1_x1, mu1[0] + 2*sigma1_x1, num = num_rbf_axis)
rbf_cl1_x2 = np.linspace(mu1[1] - 2*sigma1_x2, mu1[1] + 2*sigma1_x2, num = num_rbf_axis)



rbf_cl1 = [0, 0]

for i in range(num_rbf_axis):
    for j in range(num_rbf_axis):
        rbf_row = np.append(rbf_cl1_x1[i], rbf_cl1_x2[j])
        rbf_cl1 = np.vstack([rbf_cl1, rbf_row])

rbf_cl1 = rbf_cl1[1:len(rbf_cl1)]
a_cl1 = np.zeros((1, len(rbf_cl1))) + 1

beta = 0.2
alpha_cl1 = 0.35*0.21
alpha1 = 10
#alpha_cl1_a = alpha_cl1*gamma

phi_syn = synth_phi_mat(data_train_cl1, rbf_cl1, alpha_cl1, alpha1)
phi_test = synth_phi_mat(data_test_cl1, rbf_cl1, alpha_cl1, alpha1)
y_syn = get_y(phi_syn, a_cl1)
y_test = get_y(phi_test, a_cl1)
print(y_syn)

avg_pow_y = np.mean(y_syn**2)
avg_pow_y_db = 10*np.log10(avg_pow_y)
mean_noise = 0
SNR_db = 10

neigh_cl1 = KNeighborsRegressor(n_neighbors=4)

MAE_all_i = []
MAE_all_i_knnr = []
for i in range(len(gamma)):

    phi_ana = synth_phi_mat(data_train_cl1, rbf_cl1, gamma[i]*alpha_cl1, alpha1)
    phi_ana_test = synth_phi_mat(data_test_cl1, rbf_cl1, gamma[i]*alpha_cl1, alpha1)

    IM_ana = get_IM(phi_ana, rbf_cl1)

    diff_mean_all_j = []
    diff_mean_all_j_knnr = []
    for j in range(1000):
        print(i, j)
        noise_y_train = get_noisy_y(avg_pow_y_db, SNR_db, y_syn)
        noise_y_test = get_noisy_y(avg_pow_y_db, SNR_db, y_test)

        a_hat = get_ahat(IM_ana, phi_ana, noise_y_train)

        y_pred = get_y(phi_test, a_hat)
        diff_per_j = np.abs(y_pred-y_test)
        diff_mean_per_j = np.mean(diff_per_j)
        diff_mean_all_j = np.append(diff_mean_all_j, diff_mean_per_j)

        neigh_cl1.fit(data_test_cl1, noise_y_test)
        y_pred_knnr = neigh_cl1.predict(data_test_cl1)

        diff_per_j_knnr = np.abs(y_pred_knnr - y_test)
        diff_mean_per_j_knnr = np.mean(diff_per_j_knnr)
        diff_mean_all_j_knnr = np.append(diff_mean_all_j_knnr, diff_mean_per_j_knnr)

    diff_mean_per_i = np.mean(diff_mean_all_j)
    MAE_all_i = np.append(MAE_all_i, diff_mean_per_i)
    diff_mean_per_i_knnr = np.mean(diff_mean_all_j_knnr)
    MAE_all_i_knnr = np.append(MAE_all_i_knnr, diff_mean_per_i_knnr)

print(MAE_all_i)
plt.plot(gamma, np.log10(MAE_all_i), 'o')
plt.plot(gamma, np.log10(MAE_all_i_knnr), 'x')
plt.ylabel('Logarithm of MAE by both regression methods')
plt.xlabel('alpha_a/alpha_s')
plt.title('Performance of RBF-LSE under mismatching in 2D space')
plt.show()

