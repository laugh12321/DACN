# Hyperspectral Unmixing

Hyperspectral image unmixing of convolutional neural network (No fork, semi-finished products)

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
