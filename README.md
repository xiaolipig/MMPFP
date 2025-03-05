# **MMPFP**

- Our work and code are mainly inspired by and reference the following research (partial):
  - [Zhu et al., 2024](https://academic.oup.com/bioinformatics/article/40/10/btae571/7766190?searchresult=1)
  - [TAWFN](https://academic.oup.com/bioinformatics/article/40/10/btae571/7766190?searchresult=1)
  - [MDPI article](https://www.mdpi.com/2218-273X/12/11/1709)
  - [Nature article](https://www.nature.com/articles/s41467-021-23303-9)
  - [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-031-19836-6_20)
  - [Bioinformatics article](https://academic.oup.com/bioinformatics/article/37/18/2825/6182677?login=false)

## **Introduction**

MMPFP  is a deep learning-based framework that integrates protein sequence and structural information for function prediction. Our model combines **GCN, CNN, and Transformer** modules to enhance feature extraction and improve accuracy across **Molecular Function (MF), Biological Process (BP), and Cellular Component (CC)** classification tasks.



## **Environment Setup**

To run MMPFP, we recommend configuring your environment with **Huawei MindSpore Framework**. The required framework version is **MindSpore 2.0.0rc1**, and the firmware version should be **CANN 6.0.0**. You can use either **Ascend 910B** or a **4070 GPU** for training. For GPU, debugging interfaces may need to be enabled to ensure compatibility. The initial learning rate should be set to **0.0001**.

The required Python version is **Python 3.8**.

------

## **Requirements**

Below is a list of the primary requirements for the project :

- **mindspore** == 2.0.0rc1
- **numpy** == 1.22.3
- **pillow** == 9.5.0
- **scikit-learn** == 1.0.1
- **torch** == 1.10.0
- **biopython** == 1.79
- **matplotlib** == 3.4.3
- **pandas** == 1.3.3



For installing **MindSpore**, please visit [MindSpore official website](https://www.mindspore.cn/versions) for detailed installation instructions tailored to your environment (Ascend, GPU, or CPU).



Alternatively, you can download and use our pre-configured Docker image for an easy setup. To download the image, please visit the following [here](https://pan.baidu.com/s/1xWXl5BAVSKEKvbKCxQM_PA?pwd=h4m2) (code is h4m2).- Note: The download link is valid for 7 days. We will continuously update the links periodically, or you may contact the corresponding author directly to obtain the latest version and follow the instructions for downloading the Docker image to your local machine.

Environment Compatibility Notice:
The Docker image encapsulates our extensive refactoring work (PyTorch→MindSpore) for maximum portability. Current version guarantees stability on NVIDIA GPUs. For Ascend 910B deployment:
• Immediate use requires manual CANN 6.0.0 configuration via conda
• Optimized Ascend image will be released in subsequent versions
Contact us prior to any secondary development or commercial application.



### How to Test the Model:

To test the model, you can run the following command:

```python
python test.py --batch_size 32 --device_target GPU --ckpt_path ./checkpoints/best_model.ckpt
```

The training script follows a similar pattern to the testing script. You can adjust the parameters such as batch size, learning rate, and the path for the dataset or checkpoint in the training script to fit your needs. The key difference is that the training script will involve training your model, while the test script will perform inference with an already trained model.


