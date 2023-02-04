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
SNR_base_db = 20
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

mu2 = [300]
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

a_cl1 = np.zeros((1, len(rbf_cl1))) + 1
a_cl2 = np.zeros((1, len(rbf_cl2))) + 1


delta_cl1 = 4*sigma1/(len(rbf_cl1)-1)
delta_cl2 = 4*sigma2/(len(rbf_cl2)-1)
print(delta_cl1)
print(delta_cl2)
gamma = 1

beta = 1.6e-3*50
alpha_cl1 = beta*(1/delta_cl1)
alpha1 = 10
alpha_analysis_cl1 = gamma*alpha_cl1

beta = 1.6e-3*50
alpha_cl2 = beta*(1/delta_cl2)
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

y_syn = np.append(y_train_cl1, y_train_cl2)
avg_pow_y = np.mean(y_syn**2)
avg_pow_y_db = 10*np.log10(avg_pow_y)
mean_noise = 0
##########
MAE_list = []
k = np.linspace(1, 15, 15)
print(k)

data_test_cl1_real = [[0]]
data_test_cl1 = data_test_cl1.reshape(-1,1)
data_test_cl2_real = [[0]]
data_test_cl2 = data_test_cl2.reshape(-1,1)

print(np.shape(y_test_cl1))
print(type(y_test_cl1))

MAE_list = []
for i in range(15):
    neigh_cl1 = KNeighborsRegressor(n_neighbors=i+1)
    neigh_cl2 = KNeighborsRegressor(n_neighbors=i+1)

    neigh_cl1.fit(data_test_cl1, y_test_cl1)
    neigh_cl2.fit(data_test_cl2, y_test_cl2)
    print(np.shape(data_test_cl1))
    print(type(data_test_cl1))
    performance_list = []
    for k in range(1000):
        diff_list_cl1 = []
        diff_list_cl2 = []
        for j in range(len(data_test_cl1)):
            y_pred_cl1 = neigh_cl1.predict([data_test_cl1[j]])
            diff_per_y = np.abs(y_pred_cl1 - y_test_cl1[j])
            diff_list_cl1 = np.append(diff_list_cl1, diff_per_y)

        #diff_list_knnr = np.append(diff_list_cl1, diff_list_cl2)
        mean_diff = np.mean(diff_list_cl1)
        performance_list = np.append(performance_list, mean_diff)
    MAE = np.mean(performance_list)
    MAE_list = np.append(MAE_list, MAE)
print(MAE_list)
plt.plot(k, MAE_list)
plt.show()



#for i in range(trials_time):


