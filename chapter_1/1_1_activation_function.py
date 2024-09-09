import matplotlib.pyplot as plt 
from math import e
import numpy as np 
from scipy.stats import norm 

plt_sign = ["o-", "v-", "^-", "s-", "p-", "*-", "+-", "*-"]

def custom_Sigmoid(x):
    return 1/(1+e**(-x))

def custom_Tanh(x):
    return 2 * custom_Sigmoid(2*x) - 1

def custom_ReLU(x):
    return max(0, x)

def custom_Swish(x):
    return x / (1 + np.exp(-x))

def custom_GELU(x):
    return x * norm.cdf(x)

def custom_Leaky_ReLU(x, alpha = 0.01):
    return max(alpha * x, x)

def visualize(*func, **kwargs):
    for idx, method in enumerate(func, 0):
        sign_idx = idx % len(plt_sign)
        values = []
        inputs = [x/5. for x in list(range(-20,21,2))]
        for x in inputs:
            values.append(method(x))
        plt.plot(inputs, values, plt_sign[sign_idx], c = "k", label = method.__name__.replace("custom_",""))
    plt.legend()

    if kwargs["save_fig"]:
        plt.savefig(f'{kwargs["save_name"]}', dpi = 400)
        plt.clf()
    else:
        plt.show()

if __name__ == "__main__":

    visualize(custom_Tanh, custom_Sigmoid, custom_ReLU, save_fig = True, save_name = "activation_function1.png")
    visualize(custom_Leaky_ReLU, custom_Swish, custom_GELU, save_fig = True, save_name = "activation_function2.png")