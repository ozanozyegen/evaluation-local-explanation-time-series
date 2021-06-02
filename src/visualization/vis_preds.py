import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_preds(dataset, model, RUN_DIR, NUM_EXAMPLES=5):
    """ Plots 5 random predictions and saves it under model directory
    """
    rand_idxs = np.random.randint(0, len(dataset['test_y']), size=NUM_EXAMPLES)
    X, y = dataset['test_x'][rand_idxs], dataset['test_y'][rand_idxs]
    timesteps = list(range(-X.shape[1], 0))
    pred_steps = list(range(y.shape[1]))
    pred_y = model.predict(X)

    for idx, rand_idx in enumerate(rand_idxs):
        plt.clf()
        plt.title(f'Test idx: {rand_idx}')
        plt.plot(timesteps, X[idx, :, 0])
        plt.plot(pred_steps, pred_y[idx], 'red', label='Pred')
        plt.plot(pred_steps, y[idx], 'blue', label='True')
        plt.ylabel('Target Feature')
        plt.xlabel('Timesteps')
        plt.legend()
        plt.savefig(os.path.join(RUN_DIR, f'pred_{idx}.png'), format='png')