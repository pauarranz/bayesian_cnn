from typing import List, Union, Iterable
import math
from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import tensor_shape
import tensorflow_probability as tfp
from tensorflow.keras.initializers import HeNormal


#######
# Next, we define two helper functions that ensure that our data is in the correct format,
# one for the input and another one for the output data:
#######
def ensure_input(x, dtype, input_shape):
    """
    A function to ensure that our input is of the correct shape
    :param x:
    :param dtype:
    :param input_shape:
    :return:
    """
    x = tf.constant(x, dtype=dtype)
    call_rank = tf.rank(tf.constant(0, shape=input_shape, dtype=dtype)) + 1
    if tf.rank(x) < call_rank:
        x = tf.reshape(x, [-1, * input_shape.as_list()])
    return x


def ensure_output(y, dtype, output_dim):
    """
    A function to ensure that our output is of the correct shape
    :param y:
    :param dtype:
    :param output_dim:
    :return:
    """

    output_rank = 2
    y = tf.constant(y, dtype=dtype)
    if tf.rank(y) < output_rank:
        y = tf.reshape(y, [-1, output_dim])
    return y


class ReciprocalGammaInitializer:
    """
    A short class to initialize a gamma distribution: ReciprocalGammaInitializer.
    This distribution is used as the prior for PBP’s precision parameter λ and the noise parameter γ.
    """
    def __init__(self, alpha, beta):
        self.Gamma = tfp.distributions.Gamma(concentration=alpha, rate=beta)

    def __call__(self, shape: Iterable, dtype=None):
        g = 1.0 / self.Gamma.sample(shape)
        if dtype:
            g = tf.cast(g, dtype=dtype)

        return g


def get_mean_std_x_y(x, y):
    """
    Compute the means and standard deviations of our inputs and targets
    :param x:
    :param y:
    :return:
    """
    std_X_train = np.std(x, 0)
    std_X_train[std_X_train == 0] = 1
    mean_X_train = np.mean(x, 0)
    std_y_train = np.std(y)
    if std_y_train == 0.0:
        std_y_train = 1.0
    mean_y_train = np.mean(y)
    return mean_X_train, mean_y_train, std_X_train, std_y_train


def normalize(x, y, output_shape):
    """
    Use the means and standard deviations to normalize our inputs and targets
    :param x:
    :param y:
    :param output_shape:
    :return:
    """
    x = ensure_input(x, tf.float32, x.shape[1])
    y = ensure_output(y, tf.float32, output_shape)
    mean_X_train, mean_y_train, std_X_train, std_y_train = get_mean_std_x_y(x, y)
    x = (x - np.full(x.shape, mean_X_train)) / np.full(x.shape, std_X_train)
    y = (y - mean_y_train) / std_y_train
    return x, y


class PBPLayer(tf.keras.layers.Layer):
    """
    A class to handle our PBP layers
    """
    def __init__(self, units: int, dtype=tf.float32, *args, **kwargs):
        super().__init__(dtype=tf.as_dtype(dtype), *args, **kwargs)
        self.units = units

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.input_spec = tf.keras.layers.InputSpec(
            min_ndim=2, axes={-1: last_dim}
        )
        self.inv_sqrtV1 = tf.cast(
            1.0 / tf.math.sqrt(1.0 * last_dim + 1), dtype=self.dtype
        )
        self.inv_V1 = tf.math.square(self.inv_sqrtV1)

        over_gamma = ReciprocalGammaInitializer(6.0, 6.0)
        self.weights_m = self.add_weight(
            "weights_mean", shape=[last_dim, self.units],
            initializer=HeNormal(), dtype=self.dtype, trainable=True,
        )
        self.weights_v = self.add_weight(
            "weights_variance", shape=[last_dim, self.units],
            initializer=over_gamma, dtype=self.dtype, trainable=True,
        )
        self.bias_m = self.add_weight(
            "bias_mean", shape=[self.units],
            initializer=HeNormal(), dtype=self.dtype, trainable=True,
        )
        self.bias_v = self.add_weight(
            "bias_variance", shape=[self.units],
            initializer=over_gamma, dtype=self.dtype, trainable=True,
        )
        self.Normal = tfp.distributions.Normal(
            loc=tf.constant(0.0, dtype=self.dtype),
            scale=tf.constant(1.0, dtype=self.dtype),
        )
        self.built = True


class PBdivLULayer(PBPLayer):
    @tf.function
    def call(self, x: tf.Tensor):
        """Calculate deterministic output"""
        # x is of shape [batch, divv_units]
        x = super().call(x)
        z = tf.maximum(x, tf.zeros_like(x))  # [batch, units]
        return z

    @tf.function
    def predict(self, previous_mean: tf.Tensor, previous_variance: tf.Tensor):
        ma, va = super().predict(previous_mean, previous_variance)
        mb, vb = get_bias_mean_variance(ma, va, self.Normal)
        return mb, vb


def get_bias_mean_variance(ma, va, normal):
    variance_sqrt = tf.math.sqrt(tf.maximum(va, tf.zeros_like(va)))
    alpha = safe_div(ma, variance_sqrt)
    alpha_inv = safe_div(tf.constant(1.0, dtype=alpha.dtype), alpha)
    alpha_cdf = normal.cdf(alpha)
    gamma = tf.where(
        alpha < -30,
        -alpha + alpha_inv * (-1 + 2 * tf.math.square(alpha_inv)),
        safe_div(normal.prob(-alpha), alpha_cdf),
        )
    vp = ma + variance_sqrt * gamma
    bias_mean = alpha_cdf * vp
    bias_variance = bias_mean * vp * normal.cdf(-alpha) + alpha_cdf * va * (
            1 - gamma * (gamma + alpha)
    )
    return bias_mean, bias_variance


def network():
    """

    :return:
    """
    # create a list of all layers in our network
    units = [50, 50, 1]
    layers = []
    last_shape = X_train.shape[1]

    for unit in units[:-1]:
        layer = PBdivLULayer(unit)
        layer.build(last_shape)
        layers.append(layer)
        last_shape = unit
    layer = PBPLayer(units[-1])
    layer.build(last_shape)
    layers.append(layer)
    return layers

class PBP:
    def __init__(
        self,
        layers: List[tf.keras.layers.Layer],
        dtype: Union[tf.dtypes.DType, np.dtype, str] = tf.float32
    ):
        self.alpha = tf.Variable(6.0, trainable=True, dtype=dtype)
        self.beta = tf.Variable(6.0, trainable=True, dtype=dtype)
        self.layers = layers
        self.Normal = tfp.distributions.Normal(
            loc=tf.constant(0.0, dtype=dtype),
            scale=tf.constant(1.0, dtype=dtype),
        )
        self.Gamma = tfp.distributions.Gamma(
            concentration=self.alpha, rate=self.beta
        )

    def fit(self, x, y, batch_size: int = 16, n_epochs: int = 1):
        data = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
        for epoch_index in range(n_epochs):
            print(f"{epoch_index=}")
            for x_batch, y_batch in data:
                diff_square, v, v0 = self.update_gradients(x_batch, y_batch)
                alpha, beta = update_alpha_beta(
                    self.alpha, self.beta, diff_square, v, v0
                )
                self.alpha.assign(alpha)
                self.beta.assign(beta)

    @tf.function
    def predict(self, x: tf.Tensor):
        m, v = x, tf.zeros_like(x)
        for layer in self.layers:
            m, v = layer.predict(m, v)
        return m, v
    @tf.function
    def update_gradients(self, x, y):
        trainables = [layer.trainable_weights for layer in self.layers]
        with tf.GradientTape() as tape:
            tape.watch(trainables)
            m, v = self.predict(x)
            v0 = v + safe_div(self.beta, self.alpha - 1)
            diff_square = tf.math.square(y - m)
            logZ0 = logZ(diff_square, v0)
        grad = tape.gradient(logZ0, trainables)
        for l, g in zip(self.layers, grad):
            l.apply_gradient(g)
        return diff_square, v,

    @tf.function
    def predict(self, previous_mean: tf.Tensor, previous_variance: tf.Tensor):
        mean = (
                       tf.tensordot(previous_mean, self.weights_m, axes=[1, 0])
                       + tf.expand_dims(self.bias_m, axis=0)
               ) * self.inv_sqrtV1

        variance = (
                           tf.tensordot(
                               previous_variance, tf.math.square(self.weights_m), axes=[1, 0]
                           )
                           + tf.tensordot(
                       tf.math.square(previous_mean), self.weights_v, axes=[1, 0]
                   )
                           + tf.expand_dims(self.bias_v, axis=0)
                           + tf.tensordot(previous_variance, self.weights_v, axes=[1, 0])
                   ) * self.inv_V1

        return mean, variance

    @tf.function
    def apply_gradient(self, gradient):
        dlogZ_dwm, dlogZ_dwv, dlogZ_dbm, dlogZ_dbv = gradient

        # Weights
        self.weights_m.assign_add(self.weights_v * dlogZ_dwm)
        new_mean_variance = self.weights_v - (
                tf.math.square(self.weights_v)
                * (tf.math.square(dlogZ_dwm) - 2 * dlogZ_dwv)
        )
        self.weights_v.assign(non_negative_constraint(new_mean_variance))

        # Bias
        self.bias_m.assign_add(self.bias_v * dlogZ_dbm)
        new_bias_variance = self.bias_v - (
                tf.math.square(self.bias_v)
                * (tf.math.square(dlogZ_dbm) - 2 * dlogZ_dbv)
        )
        self.bias_v.assign(non_negative_constraint(new_bias_variance))

#########
# Loss function
#########
pi = tf.math.atan(tf.constant(1.0, dtype=tf.float32)) * 4
LOG_INV_SQRT2PI = -0.5 * tf.math.log(2.0 * pi)


@tf.function
def logZ(diff_square: tf.Tensor, v: tf.Tensor):
    v0 = v + 1e-6
    return tf.reduce_sum(
        -0.5 * (diff_square / v0) + LOG_INV_SQRT2PI - 0.5 * tf.math.log(v0)
    )


@tf.function
def logZ1_minus_logZ2(diff_square: tf.Tensor, v1: tf.Tensor, v2: tf.Tensor):
    return tf.reduce_sum(
        -0.5 * diff_square * safe_div(v2 - v1, v1 * v2)
        - 0.5 * tf.math.log(safe_div(v1, v2) + 1e-6)
    )

#############
# As discussed in the previous section, PBP belongs to the class of Assumed Density Filtering (ADF) methods.
# As such, we update the α and β parameters according to ADF’s update rules:
#############
def update_alpha_beta(alpha, beta, diff_square, v, v0):
    alpha1 = alpha + 1
    v1 = v + safe_div(beta, alpha)
    v2 = v + beta / alpha1
    logZ2_logZ1 = logZ1_minus_logZ2(diff_square, v1=v2, v2=v1)
    logZ1_logZ0 = logZ1_minus_logZ2(diff_square, v1=v1, v2=v0)
    logZ_diff = logZ2_logZ1 - logZ1_logZ0
    Z0Z2_Z1Z1 = safe_exp(logZ_diff)
    pos_where = safe_exp(logZ2_logZ1) * (alpha1 - safe_exp(-logZ_diff) * alpha)
    neg_where = safe_exp(logZ1_logZ0) * (Z0Z2_Z1Z1 * alpha1 - alpha)
    beta_denomi = tf.where(logZ_diff >= 0, pos_where, neg_where)
    beta = safe_div(beta, tf.maximum(beta_denomi, tf.zeros_like(beta)))

    alpha_denomi = Z0Z2_Z1Z1 * safe_div(alpha1, alpha) - 1.0

    alpha = safe_div(
        tf.constant(1.0, dtype=alpha_denomi.dtype),
        tf.maximum(alpha_denomi, tf.zeros_like(alpha)),
    )

    return alpha, beta

@tf.function
def safe_div(x: tf.Tensor, y: tf.Tensor, eps: tf.Tensor = tf.constant(1e-6)):
    _eps = tf.cast(eps, dtype=y.dtype)
    return x / (tf.where(y >= 0, y + _eps, y - _eps))


@tf.function
def safe_exp(x: tf.Tensor, BIG: tf.Tensor = tf.constant(20)):
    return tf.math.exp(tf.math.minimum(x, tf.cast(BIG, dtype=x.dtype)))


@tf.function
def non_negative_constraint(x: tf.Tensor):
    return tf.maximum(x, tf.zeros_like(x))


if __name__ == '__main__':
    # To make sure we produce the same output every time, we initialize our seeds:
    RANDOM_SEED = 0
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # load the California Housing dataset
    X, y = datasets.fetch_california_housing(return_X_y=True)
    # split the data (X) and targets (y) into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    # run our normalize() function on our data
    x, y = normalize(X_train, y_train, 1)

    layers = network()
    model = PBP(layers)
    model.fit(x, y, batch_size=1, n_epochs=1)

    # Compute our means and standard deviations
    mean_X_train, mean_y_train, std_X_train, std_y_train = get_mean_std_x_y(X_train, y_train)

    # Normalize our inputs
    X_test = (X_test - np.full(X_test.shape, mean_X_train)) / np.full(X_test.shape, std_X_train)

    # Ensure that our inputs are of the correct shape
    X_test = ensure_input(X_test, tf.float32, X_test.shape[1])
