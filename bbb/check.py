import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

error_normal_nn = np.loadtxt("./result/error/errors_normal_nn.txt")
error_bbb_nn = np.loadtxt("./result/error/errors_bbb_nn.txt")

plt.plot(range(len(error_normal_nn)),error_normal_nn,label="error_normal_nn")
plt.plot(range(len(error_bbb_nn)),error_bbb_nn,label="error_bbb_nn")
plt.title("Loss through iteration")
plt.legend()
plt.savefig("./result/figure/loss.png",dpi=300)
plt.close()