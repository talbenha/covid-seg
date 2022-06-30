import tensorflow as tf

def class_weighted_pixelwise_crossentropy_final(target, output, ce_weights):
    output = tf.nn.softmax(output)
    output = tf.clip_by_value(output, 10e-8, 1. - 10e-8)
    weights = tf.constant(ce_weights, dtype=tf.float32)
    maps=[tf.cast(tf.compat.v1.equal(target[..., 0], i), tf.float32) for i in range(len(ce_weights))]
    cross=[tf.math.multiply(maps[i], tf.math.log(output[..., i])) * weights[i] for i in range(len(ce_weights))]

    cross_sum=cross[0]
    for i in range(1,len(ce_weights)):
        cross_sum+=cross[i]

    loss = -tf.reduce_mean(cross_sum)
    return loss

def dice_loss(target, output, dice_weights):
    output = tf.nn.softmax(output, axis=-1)
    output = tf.clip_by_value(output, 10e-8, 1. - 10e-8)
    weights = tf.constant(dice_weights, dtype=tf.float32)

    maps = [tf.cast(tf.compat.v1.equal(target[..., 0], i), tf.float32) for i in range(len(dice_weights))]
    intersections = [tf.reduce_sum(tf.math.multiply(maps[i], output[..., i])) * weights[i] * 2 for i in range(len(dice_weights))]
    unions = [tf.reduce_sum(maps[i]) + tf.reduce_sum(output[..., i])  for i in range(len(dice_weights))]
    dices=[tf.math.divide(intersections[i] + 1, unions[i] + 1)  for i in range(len(dice_weights))]

    dice_sum = dices[0]
    for i in range(1, len(dice_weights)):
        dice_sum += dices[i]

    dice_loss=1-dice_sum

    return dice_loss

def loss_build(target, output, lam=0.5):
    return lam * class_weighted_pixelwise_crossentropy_final(target, output) + (1 - lam) * dice_loss(target, output)

