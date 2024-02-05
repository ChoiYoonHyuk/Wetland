import matplotlib.pyplot as plt
import numpy as np


def visualize(x1, x2):
    plt.ylim(-0.05, 1)
    
    t = np.arange(20, 50 * len(x1), 50)
    
    plt.plot(t, x1, 'ro--', label='Valid. Acc')
    plt.plot(t, x2, 'g^--', label='Test Acc')
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy (%)")
    #plt.title("MLP, Input -> 16 -> 5")
    plt.legend(loc='upper left')
    
    plt.savefig('mlp.png')