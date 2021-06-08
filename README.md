# DACN 

## Introduction

In this article, we design an end-to-end hyperspectral unmixing method based on dual attention convolutional neural network (DACN), which adds two types of attention modules on the basis of feature extraction by CNN, and models the semantic information on spectral-spatial dimensions to adaptively fuse local and global features. Furthermore, Layer normalization and Maxpooling are used on DACN to avoid over fitting. The evaluation of the complete performance is carried out on two hyperspectral datasets: Jasper Ridge and Urban. Compared with that of the existing method, our method can extract spectral-spatial feature information more effectively, and the precision is improved significantly.

### Network Architecture

<div align=center> 
    <img src='images/Architecture.png'>
    Fig 1. Architecture of DACN with spectral–spatial feature extraction of HSI.
</div>

## Requirement

- Python 3.8
- TensorFlow 2.3.0

Recommend use conda create a virtual environment and to install dependencies using: 
```
pip install -r requirements.txt
```

## Usage

After setting the parameters in [`config/config.json`](config/config.json), enter the following command in the terminal:

```
python run.py
```

<div align=center> 
    <img src='images/Learning%20Rate.png'>
    Fig 2. Quantitative analysis of learning rate for the DACN method in the Jasper Ridge datasets.
</div>

<b>More Details:</b>

Use `python run.py -h` to get more parameters setting details.

## Datasets

We provide two processed datasets: Jasper Ridge(jasper), Urban(urban) in [datasets](datasets).

- <b>data.npy:</b> hyperspectral data file.

- <b>data_gt.npy:</b> ground truth file.

- <b>data_m.npy:</b> endmembers file.

## Result

### Training Loss

<table>
    <tr>
        <td>
            <img src='images/Jasper%20Loss.png'>
            Fig 3. training loss of the Jasper Ridge datasets by different methods.
        </td>
        <td>
            <img src='images/Urban%20Loss.png'>
            Fig 4. training loss of the Urban datasets by different methods.
        </td>
    </tr>
</table>

### Unmixing Result

<div align=center> 
    <img src='images/rmsAAD.png'>
    Fig 5. rmsAAD values of the Jasper Ridge and Urban datasets by different methods.
</div>


<div align=center> 
    <img src='images/Estimated%20Abundances.png'>
    Fig 6. Ground-truth and estimated abundances obtained for each endmember material in the Urban datasets by different methods.
</div>

## Citation

If you find DACN useful in your research, please consider citing.

## Misc.

Code has been tested under:

- Windows 10 with 32GB memory, a RTX2060 6G GPU and AMD R7-4800H CPU.
