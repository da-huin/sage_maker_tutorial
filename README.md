<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="./static/icon.png" alt="Project logo" ></a>
 <br>

</p>

<h3 align="center">SageMaker Tutorial</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/da-huin/sage_maker_tutorial.svg)](https://github.com/da-huin/sage_maker_tutorial/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/da-huin/sage_maker_tutorial.svg)](https://github.com/da-huin/sage_maker_tutorial/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)


</div>

---

<p align="center"> 
    <br> This page is summary of [AWS SageMaker Get Started].
</p>

## 📝 Table of Contents

- [Getting Started](#getting_started)
- [Acknowledgments](#acknowledgement)

## 🏁 Getting Started <a name = "getting_started"></a>

We will use items to create a model.
* Data: MNIST Data (handwritten single digit numbers)
* Algorithm: XGBoost algorithm provided by Amazon SageMaker.

1. Create an Amazon S3 Bucket.
  * SageMaker save [The model training data] and [Model Artifacts]
  * Include sagemaker in the bucket name. For example sagemaker-helloworld.

1. Create an Notebook Instance
  
  ![](./static/notebook.png)

  1. Open the SageMaker Console at https://console.aws.amazon.com/sagemaker/
  
  1. Fill forms and create notebook instance.

1. Create a Jupyter Notebook.
  
  ![](./static/open_jupyterlab.png)

  ![](./static/conda_python3.png)

1. Download, Explore, and Transform the Training Data.

  1. To download dataset, copy and paste the following code into the notebook and run it.

  ```python
  # %%time is magic command in Ipython.
  %%time 
  import pickle, gzip, urllib.request, json
  import numpy as np

  # Load the dataset
  urllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")
  with gzip.open('mnist.pkl.gz', 'rb') as f:
      train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
  print(train_set[0].shape)
  ```

  result: 

  ```bash
  (50000, 784)
  CPU times: user 1.05 s, sys: 320 ms, total: 1.37 s
  Wall time: 22.6 s
  ```

  1. Explore the Training Dataset.

  ```python
  %matplotlib inline
  import matplotlib.pyplot as plt
  fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(10, 10))

  for i in range(0, 10):
      img = train_set[0][i]
      label = train_set[1][i]
      img_reshape = img.reshape((28,28))
      ax = axes[i]
      imgplot = ax.imshow(img_reshape, cmap='gray')
      ax.axis("off")
      ax.set_title(label)

  plt.show()
  ```

  result:
    
  ![numbers](./static/numbers.png)

  1. Transform the Training Dataset and Upload It to Amazon S3

  ```python
  %%time

  import os
  import boto3
  import re
  import copy
  import time
  import io
  import struct
  from time import gmtime, strftime
  from sagemaker import get_execution_role

  role = get_execution_role()

  region = boto3.Session().region_name

  bucket='myBucket' # Replace with your s3 bucket name
  prefix = 'sagemaker/xgboost-mnist' # Used as part of the path in the bucket where you store data

  def convert_data():
      data_partitions = [('train', train_set), ('validation', valid_set), ('test', test_set)]
      for data_partition_name, data_partition in data_partitions:
          print('{}: {} {}'.format(data_partition_name, data_partition[0].shape, data_partition[1].shape))
          labels = [t.tolist() for t in data_partition[1]]
          features = [t.tolist() for t in data_partition[0]]
          
          if data_partition_name != 'test':
              examples = np.insert(features, 0, labels, axis=1)
          else:
              examples = features
          #print(examples[50000,:])
          
          
          np.savetxt('data.csv', examples, delimiter=',')
          
          
          
          key = "{}/{}/examples".format(prefix,data_partition_name)
          url = 's3://{}/{}'.format(bucket, key)
          boto3.Session().resource('s3').Bucket(bucket).Object(key).upload_file('data.csv')
          print('Done writing to {}'.format(url))
          
  convert_data()    
  ```

## 🎉 Acknowledgements <a name = "acknowledgement"></a>

- Title icon made by [Freepik](https://www.flaticon.com/kr/authors/freepik).

- If you have a problem. please make [issue](https://github.com/da-huin/sage_maker_tutorial/issues).

- Please help develop this project 😀

- Thanks for reading 😄
