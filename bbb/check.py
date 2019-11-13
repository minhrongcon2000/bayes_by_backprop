import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

weight1_normal_nn = np.loadtxt("./result/weights/weight1_normal_nn.txt")
weight2_normal_nn = np.loadtxt("./result/weights/weight2_normal_nn.txt")
bias_normal_nn = np.loadtxt("./result/weights/bias_normal_nn.txt")
error_normal_nn = np.loadtxt("./result/weights/errors_normal_nn.txt")

weight1_bbb_nn = np.loadtxt("./result/weights/weight1_bbb_nn.txt")
weight2_bbb_nn = np.loadtxt("./result/weights/weight2_bbb_nn.txt")
bias_bbb_nn = np.loadtxt("./result/weights/bias_bbb_nn.txt")
error_bbb_nn = np.loadtxt("./result/weights/errors_bbb_nn.txt")

sns.distplot(weight1_normal_nn,hist=False,label='weight1_normal_nn')
sns.distplot(weight1_bbb_nn,hist=False,label='weight1_bbb_nn')
plt.title("Weight1 distribution")
plt.savefig("./result/figure/weight1.png",dpi=300)
plt.show()

sns.distplot(weight2_normal_nn,hist=False,label="weight2_normal_nn")
sns.distplot(weight2_bbb_nn,hist=False,label="weight2_bbb_nn")
plt.title("Weight2 distribution")
plt.savefig("./result/figure/weight2.png",dpi=300)
plt.show()

sns.distplot(bias_normal_nn,hist=False,label="bias_normal_nn")
sns.distplot(bias_bbb_nn,hist=False,label="bias_bbb_nn")
plt.title("Bias distribution")
plt.savefig("./result/figure/bias.png",dpi=300)
plt.show()

plt.plot(range(len(error_normal_nn)),error_normal_nn,label="error_normal_nn")
plt.plot(range(len(error_bbb_nn)),error_bbb_nn,label="error_bbb_nn")
plt.title("Loss through iteration")
plt.legend()
plt.savefig("./result/figure/loss.png",dpi=300)
plt.show()