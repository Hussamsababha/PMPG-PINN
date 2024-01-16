import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from lbfgs import optimizer as lbfgs_op
from matplotlib import pyplot as plt
import pdb
import time
from functools import partial


class PINNs(models.Model):
    def __init__(self, model, optimizer, epochs, **kwargs):
        super(PINNs, self).__init__(**kwargs)
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.hist = []
        self.f1 = []
        self.epoch = 0
        self.sopt = lbfgs_op(self.trainable_variables)
        self.nu = 0.01
              
    @tf.function
    def net_f(self, cp):
        cp = self.scalex_r(cp)
        x = cp[:, 0]
        y = cp[:, 1]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            X = tf.stack([x, y], axis = -1)
            X = self.scalex(X)
            pred = self.model(X)
            U = pred[:, 0]
            V = pred[:, 1]
            U = X[:, 1]*(-X[:, 0]**2 + X[:, 0]) + X[:, 1]*(1 - X[:, 1])*X[:, 0]*(1 - X[:, 0])*U
            V = X[:, 1]*(1 - X[:, 1])*X[:, 0]*(1 - X[:, 0])*V
            U_x = tape.gradient(U, x)
            U_y = tape.gradient(U, y)
            V_x = tape.gradient(V, x)
            V_y = tape.gradient(V, y)
        U_xx = tape.gradient(U_x, x)
        U_yy = tape.gradient(U_y, y)
        V_xx = tape.gradient(V_x, x)
        V_yy = tape.gradient(V_y, y)

        f1 = U_x + V_y
        f2 = (U * U_x + V * U_y - self.nu * (U_xx + U_yy))
        f3 = (U * V_x + V * V_y - self.nu * (V_xx + V_yy))

        return f1, f2, f3

    @tf.function
    def train_step(self, cp, mu, lambla):

        with tf.GradientTape() as tape:

            # Gradient Momentum
            f1, f2, f3 = self.net_f(cp)
            loss_continuity = tf.reduce_mean(tf.square(f1))
            loss_fmomentgrad = 0.5*1.0*(tf.reduce_mean((tf.square(f2)) + tf.square(f3)))
            loss_lambla = tf.reduce_mean(lambla * f1)
            loss = (1.0 * loss_fmomentgrad) - loss_lambla + (mu/2 * loss_continuity)

        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        
        l1 = tf.reduce_mean(loss)
        l2 = tf.reduce_mean(loss_continuity)
        l3 = tf.reduce_mean(loss_fmomentgrad)

        #lambda_avg = tf.reduce_mean(lambla)
        return loss, grads, tf.stack([l1, l2, l3]), f1


    # @tf.function
    def fit_scale(self, y):
        ymax = 1
        self.ymax = ymax
        return y / ymax
    
    @tf.function
    def scale(self, y):
        ymax = 1
        self.ymax = ymax
        return y / self.ymax
    
    @tf.function
    def scale_r(self, ys):
        ymax = 1
        self.ymax = ymax
        return ys * self.ymax
    
    # @tf.function
    def fit_scalex(self, x):
        # xmax = tf.reduce_max(tf.abs(x), axis = 0)
        # xmin = tf.reduce_min(x, axis = 0)
        xmax = tf.constant([1.0, 1.0], dtype=tf.float32)
        xmin = tf.constant([0.0, 0.0], dtype=tf.float32)
        self.xmax = tf.constant([1.0, 1.0], dtype=tf.float32)
        self.xmin = tf.constant([0.0, 0.0], dtype=tf.float32)
        xs = ((x - xmin) / (xmax - xmin))
        return xs
    
    @tf.function
    def scalex(self, x):
        xs = ((x - self.xmin) / (self.xmax - self.xmin)) 
        return xs
    
    @tf.function
    def scalex_r(self, xs):
        x = (xs) * (self.xmax - self.xmin) + self.xmin
        return x

    def fit(self,  cp, lambla, mu):
        cp = tf.convert_to_tensor(cp, tf.float32)
        lambla = tf.convert_to_tensor(lambla, tf.float32)
        cp = self.fit_scalex(cp)

        def func(params_1d):
            self.sopt.assign_params(params_1d)
            loss, grads, hist, f1 = self.train_step(cp, mu, lambla)
            grads = tf.dynamic_stitch(self.sopt.idx, grads)
            self.epoch += 1
            self.hist.append(hist.numpy())
            self.f1 = f1.numpy()
            if (self.epoch + 1) % 100 == 0:
                tf.print('epoch:', self.epoch)
                tf.print('hist :', hist)
            return loss.numpy().astype(np.float64), grads.numpy().astype(np.float64), f1.numpy().astype(np.float64)
        
        for epoch in range(self.epochs):
            loss, grads, hist, f1 = self.train_step(cp, mu, lambla)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            self.epoch += 1
            self.hist.append(hist.numpy())
            if (self.epoch + 1) % 100 == 0:
                tf.print('epoch:', self.epoch)
                tf.print('hist :', hist)

        # Create a new function with fixed arguments
        #partial_func = partial(func, mu=mu, lambla=lambla)

        self.sopt.minimize(func)

        # self.sopt.minimize(func)
            
        return np.array(self.hist), np.array(self.f1)
    
    def predict(self, cp):
        cp = tf.convert_to_tensor(cp, tf.float32)
        cp = self.scalex(cp)
        x = cp[:, 0]
        y = cp[:, 1]
        x = tf.expand_dims(x, axis=1)
        y = tf.expand_dims(y, axis=1)
        X = tf.concat([x, y], axis=1)
        u_p = self.model(cp)
        U = u_p[:, 0]
        V = u_p[:, 1]
        U = X[:, 1] * (-X[:, 0] ** 2 + X[:, 0]) + X[:, 1] * (1 - X[:, 1]) * X[:, 0] * (1 - X[:, 0]) * U
        V = X[:, 1] * (1 - X[:, 1]) * X[:, 0] * (1 - X[:, 0]) * V
        u_p = tf.stack([U, V], axis=1)
        u_p = self.scale_r(u_p)
        return u_p.numpy()