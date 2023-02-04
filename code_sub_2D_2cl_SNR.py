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

def get_rbf(data, mu, sigma_x1, sigma_x2, rbf_ratio):
    num_rbf = rbf_ratio * len(data)
    num_rbf_axis = int(np.round(np.sqrt(num_rbf)))
    ruozhi1 = mu[0] - 2 * sigma_x1
    ruozhi2 = mu[0] + 2 * sigma_x1
    ruozhi3 = mu[1] - 2 * sigma_x2
    ruozhi4 = mu[1] + 2 * sigma_x2
    rbf_x1 = np.linspace(ruozhi1, ruozhi2, num_rbf_axis)
    rbf_x2 = np.linspace(ruozhi3, ruozhi4, num_rbf_axis)

    print(rbf_x1)
    print(rbf_x2)
    rbf = [0, 0]
    for i in range(num_rbf_axis):
        for j in range(num_rbf_axis):
            rbf_row = np.append(rbf_x1[i], rbf_x2[j])
            rbf = np.vstack([rbf, rbf_row])

    rbf = rbf[1:len(rbf)]
    return rbf
#
trial_time = 1000
#
#
SNR_db = np.linspace(-20, 20, 41)

rbf_ratio = 0.16

data_train = genvec(2,1000)
data_test = genvec(2, 250)
plt.plot(data_train[:,0], data_train[:,1], 'o')

plt.show()

mu1 = [0, 0]
sigma1 = [50, 50]
sigma1_x1 = sigma1[0]
sigma1_x2 =sigma1[1]
sigma1 = np.diag(sigma1)

pro_cl1 = 0.55
data_train_cl1 = [0, 0]
data_test_cl1 = [0, 0]


mu2 = [500, 500]
sigma2 = [75, 75]
sigma2_x1 = sigma2[0]
sigma2_x2 = sigma2[1]
sigma2 = np.diag(sigma2)
pro_cl2 = 0.45
data_train_cl2 = [0, 0]
data_test_cl2 = [0 ,0]

for i in range(len(data_train)):
    threshold = np.random.uniform(0, 1)
    if (threshold < pro_cl1):
        data_train_cl1 = np.vstack([data_train_cl1, gencl_row(data_train[i:i + 1], mu1, sigma1)])
    else:
        data_train_cl2 = np.vstack([data_train_cl2, gencl_row(data_train[i:i + 1], mu2, sigma2)])

for i in range(len(data_test)):
    threshold = np.random.uniform(0, 1)
    if (threshold < pro_cl1):
        data_test_cl1 = np.vstack([data_test_cl1, gencl_row(data_test[i:i + 1], mu1, sigma1)])
    else:
        data_test_cl2 = np.vstack([data_test_cl2, gencl_row(data_test[i:i + 1], mu2, sigma2)])

data_train_cl1 = data_train_cl1[1:len(data_train_cl1)]
data_train_cl2 = data_train_cl2[1:len(data_train_cl2)]
data_test_cl1 = data_test_cl1[1:len(data_test_cl1)]
data_test_cl2 = data_test_cl2[1:len(data_test_cl2)]



#----syhthsis model setting
#---rbf

rbf_cl1 = get_rbf(data_train_cl1, mu1, sigma1_x1, sigma1_x2, rbf_ratio)




rbf_cl2 = get_rbf(data_train_cl2, mu2, sigma2_x1, sigma2_x2, rbf_ratio*1.2)
print(np.shape(rbf_cl2))
print(rbf_cl2)
print(rbf_cl1)
a_cl1 = np.zeros((1, len(rbf_cl1))) + 1
a_cl2 = np.zeros((1, len(rbf_cl2))) + 1


alpha_cl1 = 0.35*0.15
alpha_cl2 = 0.2*0.12
alpha1 = 10

gamma = 1
phi_cl1_syn_train = synth_phi_mat(data_train_cl1, rbf_cl1, alpha_cl1, alpha1)
phi_cl2_syn_train = synth_phi_mat(data_train_cl2, rbf_cl2, alpha_cl2, alpha1)
phi_cl1_syn_test = synth_phi_mat(data_test_cl1, rbf_cl1, alpha_cl1, alpha1)
phi_cl2_syn_test = synth_phi_mat(data_test_cl2, rbf_cl2, alpha_cl2, alpha1)

phi_cl1_ana = synth_phi_mat(data_train_cl1, rbf_cl1, gamma*alpha_cl1, alpha1)
phi_cl2_ana = synth_phi_mat(data_train_cl2, rbf_cl2, gamma*alpha_cl2, alpha1)
phi_cl1_ana_test = synth_phi_mat(data_test_cl1, rbf_cl1, gamma*alpha_cl1, alpha1)
phi_cl2_ana_test =synth_phi_mat(data_test_cl2, rbf_cl2, gamma*alpha_cl2, alpha1)

y_train_cl1 = get_y(phi_cl1_syn_train, a_cl1)
y_train_cl2 = get_y(phi_cl2_syn_train, a_cl2)
y_test_cl1 = get_y(phi_cl1_syn_test, a_cl1)
y_test_cl2 = get_y(phi_cl2_syn_test, a_cl2)


plt.plot(data_train_cl2[:,0], y_train_cl2, 'o')
plt.show()
y_train = np.append(y_train_cl1, y_train_cl2)
y_test = np.append(y_test_cl1, y_test_cl2)

IM_cl1 = get_IM(phi_cl1_ana, rbf_cl1)
IM_cl2 = get_IM(phi_cl2_ana, rbf_cl2)


print(np.linalg.cond(IM_cl1))
print(np.linalg.cond(IM_cl2))

avg_pow_y = np.mean(y_train**2)
avg_pow_y_db = 10*np.log10(avg_pow_y)
mean_noise = 0

neigh_cl1 = KNeighborsRegressor(n_neighbors=4)
neigh_cl2 = KNeighborsRegressor(n_neighbors=4)

MAE_all_i = []
MAE_all_i_knnr = []
for i in range(len(SNR_db)):
    print(SNR_db[i])

    noise_db = avg_pow_y - SNR_db[i]
    noise_watts = 10 ** (noise_db / 10)

    MAE_all_j = []
    MAE_all_j_knnr = []
    for j in range(1000):
        print(i, j)
        noise = np.random.normal(mean_noise, np.sqrt(noise_watts), len(y_train))
        noise_test = np.random.normal(mean_noise, np.sqrt(noise_watts), len(y_test))

        noise_y_train = y_train + noise
        noise_y_test = y_test + noise_test

        noise_y_train_cl1 = noise_y_train[0:len(y_train_cl1)]
        noise_y_train_cl2 = noise_y_train[len(y_train_cl1): len(y_train)]
        noise_y_test_cl1 = noise_y_test[0:len(y_test_cl1)]
        noise_y_test_cl2 = noise_y_test[len(y_test_cl1): len(y_test)]

        a_hat_cl1 = get_ahat(IM_cl1, phi_cl1_ana, noise_y_train_cl1)
        a_hat_cl2 = get_ahat(IM_cl2, phi_cl2_ana, noise_y_train_cl2)

        y_pred_LSE_cl1 = get_y(phi_cl1_ana_test, a_hat_cl1)
        y_pred_LSE_cl2 = get_y(phi_cl2_ana_test, a_hat_cl2)
        y_pred_LSE = np.append(y_pred_LSE_cl1, y_pred_LSE_cl2)

        neigh_cl1.fit(data_test_cl1, noise_y_test_cl1)
        neigh_cl2.fit(data_test_cl2, noise_y_test_cl2)

        y_pred_knnr_cl1 = neigh_cl1.predict(data_test_cl1)
        y_pred_knnr_cl2 = neigh_cl2.predict(data_test_cl2)
        y_pred_knnr = np.append(y_pred_knnr_cl1, y_pred_knnr_cl2)


        diff_LSE = np.abs(y_pred_LSE-y_test)
        diff_kNNR = np.abs(y_pred_knnr-y_test)

        mean_diff_per_j = np.mean(diff_LSE)
        mean_diff_per_j_knnr = np.mean(diff_kNNR)

        MAE_all_j = np.append(MAE_all_j, mean_diff_per_j)
        MAE_all_j_knnr = np.append(MAE_all_j_knnr, mean_diff_per_j_knnr)
    MAE_per_i = np.mean(MAE_all_j)
    MAE_all_i = np.append(MAE_all_i, MAE_per_i)
    MAE_per_i_knnr = np.mean(MAE_all_j_knnr)
    MAE_all_i_knnr = np.append(MAE_all_i_knnr, MAE_per_i_knnr)

plt.plot(SNR_db, MAE_all_i, 'o')
plt.plot(SNR_db, MAE_all_i_knnr, 'x')
plt.xlabel('SNR(db)')
plt.ylabel('MAE of kNNR(x serires) and RBF-LSE (o series)')
plt.title('Performance of kNNR and RBF-LSE VS SNR with 2 GC')
plt.show()