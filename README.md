# Blindness Detection

![alt_tag](https://miro.medium.com/max/624/0*qm-3jR10-p3xMD4S)

This project is a ongoing challange on <b>Kaggle</b> where we have to develope a strategy to design a system which will be able to detet the category of blindness in the human eye.
<br>
According to the dataset, there are following categories such as :

    0 - NO DR
    1 - MILD
    2 - MODERATE
    3 - SEVERE
    4 - PORIFERATIVE DR

Directory Structure of the dataset ( Size 10GB ):
        
    sample_submission.csv
    test.csv
    train.csv
    train.csv
    test_images.zip
    
        0005cfc8afb6.png
        00836aaacf06.png
        009c019a7309.png
        ------
        1000+ more
        
    train_images.zip
    
        000c1434d8d7.png
        00a8624548a9.png
        00cb6555d108.png
        -----
        1000+ more

For this project i will be using the <b>Google Colab</b> because this kind of heavy image dataset requires a lot powerful device and google colab provides free storage for <b>GPU </b> or <b> TPU </b> based runtime environment and <b> python</b> will be my primary language.

The next few steps will guide the user to how to setup the environment and  how to download the dataset.

Please follow the exact steps:

    1 - To install kaggle packeges and related python packages.
            ! pip install kaggle torchvision torch pandas matplotlib numpy scipy scikit-learn
    2 - Create directory name " Kaggle"
            !mkdir ~/.kaggle
    3 - Check for existence
            !ls -a /content/.kaggle
            
To use the Kaggle API, you have to create a Kaggle account. Once you have logged in, you will have to go to the ‘My Account’ section on your profile. Then you will have to click on ‘Create New API Token’ to use the Kaggle API. The ‘Create New API Token’ button will trigger a download of a file called ‘kaggle.json’. This file has the credentials of your API token for your account.

Once you have the key, please proceed further.

    4 - Svaing the credential to the file.
            import json
            token = {"username":"YOUR_USERNAME","key":"YOUR_KEY"}
            with open('/content/kaggle.json', 'w') as file:
                json.dump(token, file)
            file.close()

    5 - Creating a copy
            !cp /content/kaggle.json  ~/.kaggle/kaggle.json

    6 - Setting up the configuration.
            !kaggle config set -n path -v{/content}
    
    7 - Giving permission.
            !chmod 600 /root/.kaggle/kaggle.json

Once you finished you can check if it is working or not by the following command.

        !kaggle datasets list
If you get output like below, Great you succeeded.

        ref                                                       title                                              size  lastUpdated          downloadCount  
        --------------------------------------------------------  ------------------------------------------------  -----  -------------
    dgomonov/new-york-city-airbnb-open-data                   New York City Airbnb Open Data                      2MB  2019-08-12 16:24:45           2443  
    lakshyaag/india-trade-data                                India - Trade Data                                  1MB  2019-08-16 16:13:58           2106  
    AnalyzeBoston/crimes-in-boston                            Crimes in Boston                                   10MB  2018-09-04 17:56:03          13908  
    jolasa/waves-measuring-buoys-data-mooloolaba              Waves Measuring Buoys Data                        599KB  2019-07-07 16:59:44           1549  
    citizen-ds-ghana/health-facilities-gh                     Ghana Health Facilities                            84KB  2018-09-03 01:19:24           1239  
    doit-intl/autotel-shared-car-locations                    Shared Cars Locations                              78MB  2019-01-10 13:06:00           1546  
    ma7555/schengen-visa-stats                                Schengen Visa Stats 2017/2018                       1MB  2019-07-25 10:55:37            301  
    dareenalharthi/jamalon-arabic-books-dataset               Jamalon Arabic Books Dataset                        1MB  2019-08-15 18:58:06             72  
    samhiatt/xenocanto-avian-vocalizations-canv-usa           Avian Vocalizations from CA & NV, USA               1GB  2019-08-10 00:16:10             44  
    Madgrades/uw-madison-courses                              UW Madison Courses and Grades 2006-2017            90MB  2018-05-15 

Now, that you are all setup, let get us into the world of Deep Learning.

The Standard procedures have been considered with slightly modification in the pre-processing phase.
The steps involved for the implementation are as discussed below:

        1 - Data Loading.
        2 - Scaling and Normalization with Ben Grahams Preprocessing technique(New).
        4 - Model Loading.
        5 - Training and saving the mode.
        
<h3>1 - Data Loading</h3>
The first step is to download the dataset and load it to the environment, The dataset can be downloaded manually or usng kaggle API.To use the kaggle API ,it reuires the installation .

To download the dataset from Kaggle please use the following command:

        !kaggle competitions download -c aptos2019-blindness-detection -p /content
        
Once the dataset is downloaded create a seperate tree like directory for training and testing which will be helpfull during further structural execution.

        1. !mkdir Train_Images Test_Images
        2. !unzip train_images.zip  -d Train_Images
        3. !unzip test_images.zip  -d Test_Images/
        4. !cd Train_Images && mkdir 0 1 2 3 4
        
Once the directory structure is ready map all the potos based on the categorical data using <b> Train.csv </b>. which looks like below:

<h4> Loading csv file </h4>

        import pandas as pd
        df = pd.read_csv("/content/train.csv",engine="python")
        print(df.head())
        
<h4> Mapping each data to their respective categorie.</h4>

        import os
        for var in df.values:
            os.system("mv /content/Train_Images/"+str(var[0])+".png"+" /content/Train_Images/"+str(var[1])+"/"+str(var[0])+".png")

<h3>2 - Scaling and Normalization with Ben Grahams Preprocessing technique(New).</h3>

Now it is the time for preprocessing part , here a new strategy has been introduced whch is called as <b>Ben Graham</b> pre processing techiniques which is named after the kaggle winner of this project last year.

It has little bit of enhancement with regular preprocessing techniques but it turns out to be the great deal.The steps will be discussed as we go.

<h5><b>This is currently ongoing project</b></h5>
