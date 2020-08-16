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

Full Code is in `src/tutorial.ipynb`

#### **1. Create an Amazon S3 Bucket**

* SageMaker save [The model training data] and [Model Artifacts]
* Include sagemaker in the bucket name. For example sagemaker-helloworld.

#### **2. Create an Notebook Instance**

![](./static/notebook.png)

1. Open the SageMaker Console at https://console.aws.amazon.com/sagemaker/

1. Fill forms and create notebook instance.

  * I was choice `ml.t2.medium` but I used `ml.t2.xlarge` because of insufficient memory.

#### **3. Create a Jupyter Notebook**
  
![](./static/open_jupyterlab.png)

![](./static/conda_python3.png)

#### **4. Download, Explore, and Transform the Training Data**

1. Upgrade sagemaker version to use `sagemaker.image_uris.retrieve`

    ```bash
    !pip uninstall sagemaker -y && pip install sagemaker
    sagemaker.__version__
    ```

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

    * result: 

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

    * result: 
  
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

    bucket='sagemaker-200816' # Replace with your s3 bucket name
    prefix = 'sagemaker/xgboost-mnist' # Used as part of the path in the bucket where you store data

    s3_client = boto3.client("s3")
        
    def convert_data():
        data_partitions = [('train', train_set), ('validation', valid_set), ('test', test_set)]
        for data_partition_name, data_partition in data_partitions:
            
            print(f"{data_partition_name}: {data_partition[0].shape} {data_partition[1].shape}")
            
            labels = [t.tolist() for t in data_partition[1]]
            features = [t.tolist() for t in data_partition[0]]

            if data_partition_name != 'test':
                # examples: [[y_label, labels...], ...]
                examples = np.insert(features, 0, labels, axis=1)
            else:
                examples = features
                
            np.savetxt('data.csv', examples, delimiter=',')
                
            key = f"{prefix}/{data_partition_name}/examples"
            url = f"s3://{prefix}/{data_partition_name}"
            
            s3_client.upload_file(Filename="data.csv", Bucket=bucket, Key=key)

            print(f"Done writing to {url}")

    convert_data()
    ```

#### **5. Train a Model**

1. Choose the Training Algorithm

    * You can choise Algorithm in [built-in algorithms.](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html)

1. Create and Run a Training Job with Amazon SageMaker Python SDK

    * [Estimator Details](https://sagemaker.readthedocs.io/en/stable/estimators.html#sagemaker.estimator.Estimator)

    1. import SDK and get XGBoost Container

        ```python
        import sagemaker
        from sagemaker import image_uris

        container = image_uris.retrieve('xgboost', region=region, version="latest")
        ```
    
    1. Download the training and validation data from the S3. and set the location where you store the training output.

        ```python
        train_data_url = f"s3://{bucket}/{prefix}/train"
        validation_data_url = f"s3://{bucket}/{prefix}/validation"
        s3_output_location = f"s3://{bucket}/{prefix}/xgboost_model_sdk"

        print(train_data_url)
        ```

    1. Create an instance of the `sagemaker.estimator.Estimator`

        ```python
        xgb_model = sagemaker.estimator.Estimator(container,
                                              role, 
                                              instance_count=1, 
                                              instance_type='ml.m4.xlarge',
                                              volume_size = 5,
                                              output_path=s3_output_location,
                                              sagemaker_session=sagemaker.Session())  
        ```

    1. Set hyperparameter

        ```python
        xgb_model.set_hyperparameters(max_depth = 5,
                                  eta = .2,
                                  gamma = 4,
                                  min_child_weight = 6,
                                  silent = 0,
                                  objective = "multi:softmax",
                                  num_class = 10,
                                  num_round = 10)
        ```

    1. Create the training channels to use for the training job.

        ```python
        train_channel = sagemaker.inputs.TrainingInput(train_data_url, content_type='text/csv')
        valid_channel = sagemaker.inputs.TrainingInput(validation_data_url, content_type='text/csv')

        data_channels = {'train': train_channel, 'validation': valid_channel}
        ```

    1. To start model training, call the estimator's fit method.

        ```python
        xgb_model.fit(inputs=data_channels, logs=True)
        ```

        * result:

            ```bash
            2020-08-16 01:03:52 Starting - Starting the training job...
            2020-08-16 01:03:54 Starting - Launching requested ML instances......
            2020-08-16 01:04:59 Starting - Preparing the instances for training...
            2020-08-16 01:05:51 Downloading - Downloading input data......

            ...

            01:08:01] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 60 extra nodes, 2 pruned nodes, max_depth=5
            [01:08:02] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 50 extra nodes, 6 pruned nodes, max_depth=5
            [9]#011train-merror:0.06942#011validation-merror:0.0773

            2020-08-16 01:08:11 Uploading - Uploading generated training model
            2020-08-16 01:08:11 Completed - Training job completed
            Training seconds: 140
            Billable seconds: 140        

1. You can check your training in [Sagemaker Console.](https://console.aws.amazon.com/sagemaker/)

    ![training.png](./static/training.png)

    ![training_2.png](./static/training_2.png)

    ![training_3.png](./static/training_3.png)

#### **6. Deploy the Model to Amazon SageMaker**

To get predictions, deploy your model. The method you use depends on how you want to generate inferences:

  1. To get one inference at a time in real time, set up a persistent endpoint using Amazon SageMaker hosting services.

  2. To get inferences for an entire dataset, use Amazon SageMaker batch transform.




## 🎉 Acknowledgements <a name = "acknowledgement"></a>

- Title icon made by [Freepik](https://www.flaticon.com/kr/authors/freepik).

- If you have a problem. please make [issue](https://github.com/da-huin/sage_maker_tutorial/issues).

- Please help develop this project 😀

- Thanks for reading 😄
