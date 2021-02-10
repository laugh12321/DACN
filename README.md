# Channel Attention Hyperspectral Unmixing

Hyperspectral image unmixing of convolutional neural network based on channel attention.

## Descriptions

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

<div align="center">
    <img src="https://github.com/laugh12321/Hyperspectral-Imagery-Unmixing/blob/main/figures/CA.png" />
</div> 


```python
class Channel_attention(tf.keras.layers.Layer):

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

## Prerequisites

- Python 3.8
- TensorFlow 2.3.0

Recommend use conda create a virtual environment and to install dependencies using: `pip install -r requirements.txt`

## Usage

After setting the parameters in [`config/config.json`](https://github.com/laugh12321/Hyperspectral-Imagery-Unmixing/blob/main/config/config.json), enter the following command in the terminal:

```
python run.py
```

<b>More Details:</b>

Use `python run.py -h` to get more parameters setting details.

## Dataset

We provide two processed datasets: Jasper Ridge(jasper), Urban(urban) in datasets/

<b>data.npy:</b> hyperspectral data file.

<b>data_gt.npy:</b> ground truth file.

<b>data_m.npy:</b> endmembers file.



Update: Feb 10, 2021