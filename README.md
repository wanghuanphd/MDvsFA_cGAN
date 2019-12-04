# MDvsFA_cGAN
The tensorflow and pytorch implementations of the MDvsFA_cGAN model which is proposed in ICCV2019 paper "Huan Wang, Luping Zhou and Lei Wang. Miss Detection vs. False Alarm: Adversarial Learing for Small Object Segmentation in Infrared Images. International Conference on Computer Vision, Oct.27-Nov.2,2019. Seoul, Republic of Korea".

File and folder description: 
1) The file demo_MDvsFA_tensorflow.py is the code file of tensorflow version and 'demo_MDvsFA_pytorch.py' is the code file of pytorch version. Just run them as 'python demo_MDvsFA_tensorflow.py' or 'python demo_MDvsFA_pytorch.py'.  File path and key parameters can be tuned in the respective files.
2) The 'data' folder includes the training set in the sub-folder 'training' (both the original and ground-truth images), the test set in the sub-folder 'test_org' (only orginial images) and folder 'test_gt' (only ground-truth images). Due to size limitation when uploading each file, we separate the whole data folder into six parts (.rar files). If you install WinRAR, the six files can be automatically recognized and merged.  
3) Please manually consruct an empty folder named 'trained model' to store all generated model files and segmentation results. Please put the code file (either demo_MDvsFA_tensorflow.py or demo_MDvsFA_pytorch.py), folder 'trained model' and folder 'data' in the same directory.

Should you have any questions on the code file 'demo_MDvsFA_tensorflow.py' and the dataset, please contact Huan Wang (wanghuanphd@njust.edu.cn) and Lei Wang (leiw@uow.edu.au). Should you have any questions on the code file 'demo_MDvsFA_pytorch.py', please contact Ziyu Li (liziyu@seu.edu.cn) and Wankou Yang (wkyang@seu.edu.cn).

Finally, we sincerely acknowledge Ziyu Li and Wankou Yang from the Southeast University, People Republic of China to kindly provide the pytorch implementation of the MDvsFA_cGAN model. 
