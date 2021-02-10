# Channel Attention Hyperspectral Unmixing

Hyperspectral image unmixing of convolutional neural network based on channel attention

## Description

<table>
    <tr>
        <td>
            <img src=https://github.com/laugh12321/Hyperspectral-Imagery-Unmixing/blob/main/figures/Jasper%20Ridge.png>
        </td>
        <td>
            <img src=https://github.com/laugh12321/Hyperspectral-Imagery-Unmixing/blob/main/figures/Urban.png>
        </td>
    </tr>
</table>

### Channel Attention Layer

<center><img src=https://github.com/laugh12321/Hyperspectral-Imagery-Unmixing/blob/main/figures/CA.png align="middle" /></center>

```python
class Channel_attention(tf.keras.layers.Layer):
    """
    3D implementation of Channel attention:
    Fu, Jun, et al. "Dual attention network for scene segmentation."
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
    """

    def __init__(self):
        super(Channel_attention, self).__init__()

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma',
                                     shape=1,
                                     initializer='zeros',
                                     trainable=True)

    def call(self, inputs):
        input_shape = inputs.get_shape().as_list()

        proj_query = tf.keras.layers.Reshape((input_shape[1] * input_shape[2] * input_shape[3],
                                              input_shape[4]))(inputs)
        proj_key = tf.keras.backend.permute_dimensions(proj_query, (0, 2, 1))
        energy = tf.keras.backend.batch_dot(proj_query, proj_key)
        attention = tf.keras.activations.softmax(energy)

        outputs = tf.keras.backend.batch_dot(attention, proj_query)
        outputs = tf.keras.layers.Reshape((input_shape[1], input_shape[2], input_shape[3],
                                           input_shape[4]))(outputs)
        outputs = tf.keras.layers.multiply([self.gamma, outputs])
        outputs = tf.keras.layers.add([outputs, inputs])

        return outputs
```

## Requirements

- Python 3.8
- TensorFlow 2.3.0

