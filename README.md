# MDvsFA_cGAN
A tensorflow and pytorch implementation of the MDvsFA_cGAN model which is proposed in ICCV2019 paper "Huan Wang, Luping Zhou and Lei Wang. Miss Detection vs. False Alarm: Adversarial Learing for Small Object Segmentation in Infrared Images. International Conference on Computer Vision, Oct.27-Nov.2,2019. Seoul, Republic of Korea".
File and folder description: 
1) The file demo_MDvsFA_release_v1.py is code file of tensorflow version and 'demo_MDvsFA_pytorch.py' is the code file of pytorch version. Just run them as 'python demo_MDvsFA_tensorflow.py' or 'python demo_MDvsFA_pytorch.py'.  File path and key parameters can be tuned in the respective files.
2) The 'data' folder includes the training set in the sub-folder 'training' (both the original and ground-truth images), the test set in the sub-folder 'test_org' (the orginial images) and folder 'test_gt' (the ground-truth images).
3) Please manually consruct an empty folder named 'trained model' to store all generated model files and segmentation results. Please put demo_MDvsFA_tensorflow.py, demo_MDvsFA_tensorflow.py, folder 'trained model' and folder 'data' in the same directory.

Should you have any questions on the tensorflow version, please contact Huan Wang (wanghuanphd@njust.edu.cn) and Lei Wang (leiw@uow.edu.au).
Should you have any questions on the pytorch version, please contact Ziyu Li (liziyu@seu.edu.cn) and Wankou Yang (wkyang@seu.edu.cn).

Finally, we sincerely acknowledge Ziyu Li and Wankou Yang from the Southeast University, People Republic of China to kindly provide the pytorch implementation of the MDvsFA_cGAN model. 
