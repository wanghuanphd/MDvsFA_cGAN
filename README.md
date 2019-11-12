# MDvsFA_cGAN
A tensorflow implementation of the MDvsFA_cGAN model which is proposed in ICCV2019 paper "Huan Wang, Luping Zhou, Lei Wang. Miss Detection vs. False Alarm: Adversarial Learing for Small Object Segmentation in Infrared Images. International Conference on Computer Vision, Oct.27-Nov.2,2019. Seoul, Republic of Korea".
File and folder description: 
1) The file demo_MDvsFA_release_v1.py is the only code file. Just run it as 'python demo_MDvsFA_release_v1.py'.  File Path and key parameters can be tuned in the file.
2) The 'data' folder includes the training set in the folder 'training' and the test set in the folder 'test_org'(orginial images) and folder 'test_gt'(Ground truth).
3) Please manually consruct a NULL folder named 'trained model' to save all generated model files and segmentation results.
Please put demo_MDvsFA_release_v1.py, folder 'trained model' and folder 'data' in the same directory.
