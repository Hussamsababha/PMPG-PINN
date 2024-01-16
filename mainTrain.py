import numpy as np
from tensorflow.keras import models, layers, optimizers, activations
from tensorflow.keras.utils import plot_model
from PINN_PMPG import PINNs
from matplotlib import pyplot as plt
from time import time

# The code runs for multiple values of mu_c. Keep beta = 1
initial_mu_constraint = [10.0]
beta_values = [1.0]
for Mu in initial_mu_constraint:
    for beta in beta_values:
        mu = Mu
        ny = 300
        mu_m = 1.0
        #############################################
        #### Generate and plot Collocation points ####
        #############################################
        xmin, xmax = 0, 1.0
        ymin, ymax = 0, 1.0
        x = np.linspace(xmin, xmax, ny+1)
        y = np.linspace(ymin, ymax, ny+1)
        X, Y = np.meshgrid(x, y)
        X_flatten = X.reshape(-1, 1)
        Y_flatten = Y.reshape(-1, 1)
        cp_all = np.column_stack((X_flatten, Y_flatten))
        mask_xmin = cp_all[:, 0] > xmin
        mask_xmax = cp_all[:, 0] < xmax
        mask_ymin = cp_all[:, 1] > ymin
        mask_ymax = cp_all[:, 1] < ymax
        cp = cp_all[mask_xmax & mask_xmin & mask_ymin & mask_ymax]
        plt.scatter(cp[:, 0], cp[:, 1], color='blue', label='PDE points')
        plt.show()
        n_cp = len(cp)

        #############################################
        #### Network Hyper-parameters  ##############
        #############################################
        act = "tanh"
        nn = 20
        nl = 8
        n_adam = 1000
        outer_iter = 3
        test_name = f'_{nn}_{nl}_{act}_{n_adam}_{n_cp}_{mu}_{beta}'

        #################
        # Compiling Model
        #################
        inp = layers.Input(shape = (2 ,))
        hl = inp
        for i in range(nl):
            hl = layers.Dense(nn, activation = act)(hl)
        out = layers.Dense(2)(hl)
        model = models.Model(inp, out)
        plot_model(model, show_shapes=True)
        print(model.summary())
        lr = 1e-3
        opt = optimizers.Adam(lr)
        pinn = PINNs(model, opt, n_adam)

        #################
        # Training Process
        #################
        print(f"INFO: Start training case_{test_name}")
        st_time = time()
        #initialize Lambda.
        lambla = np.zeros((cp.shape[0]))
        g = np.zeros((cp.shape[0]))
        all_hist = []
        all_mu = []
        all_lambla = []
        all_g = []
        for _ in range(outer_iter):
            print("new mu is ", mu)
            pinn = PINNs(model, opt, n_adam)
            hist, g = pinn.fit(cp, lambla, mu)
            all_hist.append(hist)
            all_mu.append(mu)
            all_lambla.append(lambla)
            all_g.append(g)
            #update lambda
            lambla = lambla - mu * g
            mu = beta * mu
            lr = 1e-3
            opt = optimizers.Adam(lr)
            n_adam = 1000
            print(" ---------------------  ")
            print(" OuterLoop  ")
            print("WEIGHTS ARE: ")
            print("mu = ", mu)
            print("lambla = ", lambla)

        combined_hist = np.concatenate(all_hist, axis=0)
        combined_lambla = np.vstack(all_lambla)

        en_time = time()
        comp_time = en_time - st_time

        #################
        # Prediction
        ################
        xmin, xmax = 0, 1.0
        ymin, ymax = 0, 1.0
        x = np.linspace(xmin, xmax, 501)
        y = np.linspace(ymin, ymax, 501)
        X, Y = np.meshgrid(x, y)
        cpp = np.array([X.flatten(), Y.flatten()]).T
        pred = pinn.predict(cpp)
        u_p = pred[:, 0].reshape(X.shape)
        v_p = pred[:, 1].reshape(X.shape)
        pred = np.stack((u_p, v_p))

        #################
        # Save prediction
        # #################
        np.savez_compressed('pred/Test' + test_name, pred=pred, x=X, y=Y, all_hist=combined_hist, all_mu=all_mu, all_f1=all_g, ct=comp_time, cp=cp, all_lambla= all_lambla)
        print("INFO: Prediction  have been saved!")
        # Define the figure
        fig, axs = plt.subplots(figsize=(5, 5))
        prediction = pred
        diffU = prediction[0, :, :]
        diffV = prediction[1, :, :]
        heatmap = axs.pcolormesh(X, Y, np.sqrt(diffU**2 + diffV**2), cmap='jet')
        colorbar = plt.colorbar(heatmap)
        axs.set_title('Velocity Magnitude')
        plt.savefig('flow field.jpg')
        plt.show(block=False)  # block=False

        # fig, ax = plt.subplots(figsize=(8, 6))
        # epochs = np.arange(0, hist.shape[0])
        # ax.semilogy(epochs, all_hist[:, 1], color='red', label='Continuity constraint, Lc')
        # ax.semilogy(epochs, all_hist[:, 2], color='blue', label='Quantity S')
        # ax.set_ylim(bottom=1e-7)
        # # Add labels and legend to the plot
        # plt.xlabel('Iterations (epochs)')
        # plt.ylabel('Residual loss functions')
        # plt.legend()
        # plt.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.92, wspace=0.3, hspace=0.3)
        # # fig.suptitle('Time = {:.2f}'.format(comp_time, 's'))
        # plt.savefig('Residual loss functions.jpg')
        # plt.show(block=False)  # block=False
