from __future__ import division
import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.image as image
import numpy as np
from skimage import measure
from skimage.morphology  import disk,dilation
from skimage.transform  import resize

max_epoch_num = 30
max_test_num = 12000  
mini_batch_size = 10
NO_USE_NORMALIZATION = 0  # 1-just divided by 255; 0-apart from being divided by 255, also stretched to [-1,1]   
is_training = True
max_patch_num = 140000
trainImageSize = 128
ReadColorImage=1 
isJointTrain = False  #update G1 and G2 separately(False) or jointly(True).
lambda1 = 100
lambda2 = 1
task="./trained_model/MDvsFAnet"   #path of the saved model

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def lrelu(x):
    return tf.maximum(x*0.2,x)

def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        #array = np.zeros(shape, dtype=float)
        array = np.random.normal(0,0.0025,size=shape)
        print(array.shape)
        cx, cy = shape[0]//2, shape[1]//2
        c1 = shape[2]
        c2 = shape[3]
        if c1==c2 :
           for i in range(c1):
               array[cx, cy, i, i] = 1
        else :
           if c1>c2:
              cmax = c1
              for j in range(c2):
                 array[cx, cy, np.int16(j*c1/c2), j] = cmax/c2
           else :
              cmax = c2 
              for i in range(c1):
                 array[cx, cy, i, np.int16(i*c2/c1)] = cmax/c2
        return tf.constant(array, dtype=dtype)
    return _initializer

def nm(x):
    w0=tf.Variable(1.0,name='w0')
    w1=tf.Variable(0.0,name='w1')
    return w0*x+w1*slim.batch_norm(x) 

def bm(x):
    return slim.batch_norm(x)

def discriminator(input):
    with tf.variable_scope("discriminator")  as scope:
        # sub-network I
        net1 = max_pool_2x2(input)
        net1 = max_pool_2x2(net1)
        net = slim.conv2d(net1,24,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=bm,scope='d_conv1') 
        net = slim.conv2d(net,24,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=bm,scope='d_conv2') 
        net = slim.conv2d(net,24,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=bm,scope='d_conv3') 
        net_featMap=slim.conv2d(net,1,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=bm,scope='d_conv4') 
        net = tf.reshape(net_featMap,[-1,(trainImageSize*trainImageSize//16)])   
        W_fc1 = weight_variable([(trainImageSize*trainImageSize//16),128])
        b_fc1 = bias_variable([128])
        net  = tf.matmul(net,W_fc1)+b_fc1        
        net  = tf.tanh(tf.layers.batch_normalization(net,True))      
        W_fc2 = weight_variable([128,64])
        b_fc2 = bias_variable([64])
        net  = tf.matmul(net,W_fc2)+b_fc2        
        net  = tf.tanh(tf.layers.batch_normalization(net,True))       
        W_fc3 = weight_variable([64,3])
        b_fc3 = bias_variable([3])
        output_sub1 = tf.nn.softmax(tf.matmul(net,W_fc3)+b_fc3)
        realscore0,realscore1,realscore2 = tf.split(output_sub1,[mini_batch_size,mini_batch_size,mini_batch_size],0)
        # sub-network II 
        feat0,feat1,feat2 = tf.split(net_featMap,[mini_batch_size,mini_batch_size,mini_batch_size],0)
        featDist = tf.reduce_mean(tf.square(feat1-feat2))
        return realscore0,realscore1,realscore2,featDist      

def Generator1_CAN8(input):
    chn=64
    with tf.variable_scope("generator_detect")  as scope :
        net=slim.conv2d(input,chn,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g1_conv0')
        net=slim.conv2d(net,chn,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g1_conv1')  # add one more layer
        net=slim.conv2d(net,2*chn,[3,3],rate=2,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g1_conv2')
        net=slim.conv2d(net,4*chn,[3,3],rate=4,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g1_conv3')
        net=slim.conv2d(net,8*chn,[3,3],rate=8,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g1_conv4')
        net=slim.conv2d(net,4*chn,[3,3],rate=4,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g1_conv5')
        net=slim.conv2d(net,2*chn,[3,3],rate=2,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g1_conv6')
        net=slim.conv2d(net,chn,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g1_conv7')
        net=slim.conv2d(net,1,[1,1],rate=1,activation_fn=None,scope='g1_conv_last')
        return net

def Generator2_UCAN64(input,useSkip):
    with tf.variable_scope("generator_alarm")  as scope :
       chn=64       
       if useSkip :  
          net1=slim.conv2d(input,chn,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g2_conv1')
          net2=slim.conv2d(net1,chn,[3,3],rate=2,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g2_conv2')
          net3=slim.conv2d(net2,chn,[3,3],rate=4,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g2_conv3')
          net4=slim.conv2d(net3,chn,[3,3],rate=8,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g2_conv4')
          net5=slim.conv2d(net4,chn,[3,3],rate=16,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g2_conv5')
          net6=slim.conv2d(net5,chn,[3,3],rate=32,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g2_conv6')
          net7=slim.conv2d(net6,chn,[3,3],rate=64,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g2_conv7')
          net8=slim.conv2d(net7,chn,[3,3],rate=32,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g2_conv8')
          net = tf.concat([net6,net8],3)
          net9=slim.conv2d(net,chn,[3,3],rate=16,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g2_conv9')
          net = tf.concat([net5,net9],3)
          net10=slim.conv2d(net,chn,[3,3],rate=8,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g2_conv10')
          net = tf.concat([net4,net10],3)
          net11=slim.conv2d(net,chn,[3,3],rate=4,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g2_conv11')
          net = tf.concat([net3,net11],3)
          net12=slim.conv2d(net,chn,[3,3],rate=2,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g2_conv12')
          net = tf.concat([net2,net12],3)
          net13=slim.conv2d(net,chn,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g2_conv13')
          net=slim.conv2d(net13,1,[1,1],rate=1,activation_fn=None,scope='g2_conv_last')
       else :
          net1=slim.conv2d(input,chn,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g2_conv1')
          net2=slim.conv2d(net1,chn,[3,3],rate=2,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g2_conv2')
          net3=slim.conv2d(net2,chn,[3,3],rate=4,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g2_conv3')
          net4=slim.conv2d(net3,chn,[3,3],rate=8,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g2_conv4')
          net5=slim.conv2d(net4,chn,[3,3],rate=16,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g2_conv5')
          net6=slim.conv2d(net5,chn,[3,3],rate=32,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g2_conv6')
          net7=slim.conv2d(net6,chn,[3,3],rate=64,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g2_conv7')
          net8=slim.conv2d(net7,chn,[3,3],rate=32,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g2_conv8')
          net9=slim.conv2d(net8,chn,[3,3],rate=16,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g2_conv9')
          net10=slim.conv2d(net9,chn,[3,3],rate=8,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g2_conv10')
          net11=slim.conv2d(net10,chn,[3,3],rate=4,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g2_conv11')
          net12=slim.conv2d(net11,chn,[3,3],rate=2,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g2_conv12')
          net13=slim.conv2d(net12,chn,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g2_conv13')
          net=slim.conv2d(net13,1,[1,1],rate=1,activation_fn=None,scope='g2_conv_last')
    return net

def calculateF1Measure(output_image,gt_image,thre):
    output_image = np.squeeze(output_image)
    gt_image = np.squeeze(gt_image)
    out_bin = output_image>thre
    gt_bin = gt_image>thre
    recall = np.sum(gt_bin*out_bin)/np.maximum(1,np.sum(gt_bin))
    prec   = np.sum(gt_bin*out_bin)/np.maximum(1,np.sum(out_bin))
    F1 = 2*recall*prec/np.maximum(0.001,recall+prec)
    return F1

def prepare_data(task):
        input_names=[]
        output_names=[]
        train_input_names = []
        train_output_names = []
        val_names=[]
        val_gt_names=[]
   
        read_sample_num = 0
        
        #Read training samples        
        for i in range(0,10000):
            read_sample_num = read_sample_num  + 1
            input_names.append("./data/training/%06d_1.png"%(i))
            output_names.append("./data/training/%06d_2.png"%(i)) 
        #Read validation samples
        for i in range(0,100):
            filename =  "./data/test_org/%05d.png"%(i)#test_additional
            if not os.path.isfile(filename):
                   continue
            train_input_names.append(filename) #training org
            filename =  "./data/test_gt/%05d.png"%(i)
            if not os.path.isfile(filename):
                   continue
            train_output_names.append(filename) #gt
        #Read test sample, which is same to the validation set in this demo.
        for i in range(0,100):
            filename =  "./data/test_org/%05d.png"%(i)
            if not os.path.isfile(filename):
                   continue
            val_names.append(filename) #testing org
            filename =  "./data/test_gt/%05d.png"%(i)
            if not os.path.isfile(filename):
                   continue
            val_gt_names.append(filename) #gt
        return read_sample_num,input_names,output_names,val_names,train_input_names,train_output_names,val_gt_names



sess=tf.Session()

read_sample_num,input_names,output_names,val_names,train_input_names,train_output_names,val_gt_names=prepare_data(task)

gen_input=tf.placeholder(tf.float32,shape=[None,None,None,1])
gen_output=tf.placeholder(tf.float32,shape=[None,None,None,1])
dLR=tf.placeholder(tf.float32,name='d_learning_rate')
gLR=tf.placeholder(tf.float32,name='g_learning_rate')



minVar = tf.constant(0.0,dtype=tf.float32)
maxVar = tf.constant(1.0,dtype=tf.float32)
#loss for the genearator 1
gen_result1 = Generator1_CAN8(gen_input)
gen_result1 = tf.minimum(maxVar,tf.maximum(minVar,gen_result1))
MD1 = tf.reduce_mean(tf.multiply(tf.square(gen_result1-gen_output),gen_output))
FA1 = tf.reduce_mean(tf.multiply(tf.square(gen_result1-gen_output),1-gen_output))
MF_loss1 = lambda1*MD1 + FA1 

#loss for the genearator 2
gen_result2 = Generator2_UCAN64(gen_input,True)   
gen_result2 = tf.minimum(maxVar,tf.maximum(minVar,gen_result2))
MD2 = tf.reduce_mean(tf.multiply(tf.square(gen_result2-gen_output),gen_output))  
FA2 = tf.reduce_mean(tf.multiply(tf.square(gen_result2-gen_output),1-gen_output))
MF_loss2 = MD2 + lambda2*FA2 


#Here we use the average operator to integrate the results of G1 and G2, and more effective integration operators can be used.
fusion_result = (gen_result1+gen_result2)/2

pos1 = tf.concat([gen_input,2*gen_output-1],3)
neg1 = tf.concat([gen_input,2*gen_result1-1],3)
neg2 = tf.concat([gen_input,2*gen_result2-1],3)
disc_input  = tf.concat([pos1,neg1,neg2],0)
logits_real,logits_fake1,logits_fake2,Lgc = discriminator(disc_input)
disc_res = tf.concat([logits_real,logits_fake1,logits_fake2],0)


const1 = tf.constant(1.0,shape=[1,mini_batch_size],dtype=tf.float32)
const0 = tf.constant(0.0,shape=[1,mini_batch_size],dtype=tf.float32)

gen_gt  = tf.concat([const1,const0,const0],0)
gen_gt1 = tf.concat([const0,const1,const0],0)
gen_gt2 = tf.concat([const0,const0,const1],0)

gen_GC_loss  = Lgc 

ES0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real,labels=tf.transpose(gen_gt))) 
ES1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake1,labels=tf.transpose(gen_gt1))) 
ES2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake2,labels=tf.transpose(gen_gt2))) 
disc_loss = ES0 + ES1 + ES2 

gen_adv_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake1,labels=tf.transpose(gen_gt))) 
gen_adv_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake2,labels=tf.transpose(gen_gt))) 
gen_loss1  = 100*MF_loss1 + 10*gen_adv_loss1 + 1*gen_GC_loss
gen_loss2  = 100*MF_loss2 + 10*gen_adv_loss2 + 1*gen_GC_loss


G_vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'generator_detect')
G_vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'generator_alarm')
D_vars  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'discriminator')

if isJointTrain :  #combine G1 and G2 and jointly train them 
   print("joint\n")
   gen_opt  = tf.train.AdamOptimizer(learning_rate=gLR,beta1=0.5).minimize(gen_loss1+gen_loss2,var_list=G_vars1+G_vars2)
else :   #train G1 and G2 separately
   print("split\n") 
   gen_opt1 = tf.train.AdamOptimizer(learning_rate=gLR,beta1=0.9).minimize(gen_loss1,var_list=G_vars1)
   gen_opt2 = tf.train.AdamOptimizer(learning_rate=gLR,beta1=0.9).minimize(gen_loss2,var_list=G_vars2)
   #gen_opt1 = tf.train.RMSPropOptimizer(learning_rate=gLR).minimize(gen_loss1,var_list=G_vars1)
   #gen_opt2 = tf.train.RMSPropOptimizer(learning_rate=gLR).minimize(gen_loss2,var_list=G_vars2)


disc_opt = tf.train.AdamOptimizer(learning_rate=dLR,beta1=0.5).minimize(disc_loss,var_list=D_vars)

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

ckpt=tf.train.get_checkpoint_state(task)
if ckpt:
   print('loaded '+ckpt.model_checkpoint_path)
   saver.restore(sess,ckpt.model_checkpoint_path)

if is_training:
    input_images  = np.zeros((max_patch_num,trainImageSize,trainImageSize,1))
    output_images = np.zeros((max_patch_num,trainImageSize,trainImageSize,1))
    print(read_sample_num)
 
    id = 0
    id_bk = 0
    for i in range(0,read_sample_num):      
         #read ground truth images
         if not os.path.isfile(output_names[i]):
            print("cannot read the file %s\n"%(output_names[i]))
         print(i)
         bufImg = cv2.imread(output_names[i],-1)
         bufImg = np.float32(bufImg)/255.0
         output_images[id,:,:,:] = np.expand_dims(bufImg,axis=2)
         tarPixNum = np.sum(output_images[id,:,:,:])
         print("output sum=%f" % tarPixNum)
         #read input images
         if not os.path.isfile(input_names[i]):
            print("cannot read the file %s\n"%(input_names[i]))
            continue 
         real_input = np.float32(cv2.imread(input_names[i],-1))/255.0
         if  ReadColorImage==0:
              input_images[id,:,:,:] = np.expand_dims(real_input,axis=2)*2-1
         else:
              input_images[id,:,:,:] = np.expand_dims(real_input[:,:,2],axis=2)*2-1
         if id<max_patch_num :
            id += 1
    valid_sample_num = id  
    print(valid_sample_num)
    print(input_images.shape)
    print("\nbegin training...mini_batch_size=%d\n"%(mini_batch_size))
    gen_input_batch  = np.zeros((mini_batch_size,trainImageSize,trainImageSize,1))
    gen_output_batch = np.zeros((mini_batch_size,trainImageSize,trainImageSize,1))  
    
    miniBatchNum = valid_sample_num//mini_batch_size

    g_lr = np.ones((max_epoch_num))*1e-4
    d_lr = np.ones((max_epoch_num))*1e-5
    for epoch in range(1,max_epoch_num): 
        idx = np.random.permutation(valid_sample_num)
        for mbIdx in range(0,miniBatchNum):
            mbIdx1  = mbIdx * mini_batch_size
            mbIdx2  = mbIdx1 + mini_batch_size
            #==================================fill current mini-batch =====================================#       
            j=0
            for i in idx[mbIdx1:mbIdx2]:
                gen_input_batch[j,:,:,0]  = np.expand_dims(input_images[i,:,:,0],axis=0)
                gen_output_batch[j,:,:,0] = np.expand_dims(output_images[i,:,:,0],axis=0)
                j+=1
            #==================================Train discriminator =========================================#
            _,res_disc_loss,real_logits,fake_logits1,fake_logits2,res_GC_loss = sess.run([disc_opt,disc_loss,logits_real,logits_fake1,logits_fake2,gen_GC_loss],feed_dict={gen_input:gen_input_batch,gen_output:gen_output_batch,dLR:d_lr[epoch],gLR:g_lr[epoch]})
            #==================================Train generator 1 & 2 =======================================#
            if isJointTrain :
               _,res_gen_loss1,res_gen_loss2,res_MF_loss1,res_MF_loss2,res_MD1,res_FA1,res_MD2,res_FA2,res_adv_loss1,res_adv_loss2 = sess.run([gen_opt,gen_loss1,gen_loss2,MF_loss1,MF_loss2,MD1,FA1,MD2,FA2,gen_adv_loss1,gen_adv_loss2],feed_dict={gen_input:gen_input_batch,gen_output:gen_output_batch,dLR:d_lr[epoch],gLR:g_lr[epoch]})
            else :
               _,res_gen_loss1,res_MF_loss1,res_MD1,res_FA1,res_adv_loss1= sess.run([gen_opt1,gen_loss1,MF_loss1,MD1,FA1,gen_adv_loss1],feed_dict={gen_input:gen_input_batch,gen_output:gen_output_batch,dLR:d_lr[epoch],gLR:g_lr[epoch]})
               _,res_gen_loss2,res_MF_loss2,res_MD2,res_FA2,res_adv_loss2 = sess.run([gen_opt2,gen_loss2,MF_loss2,MD2,FA2,gen_adv_loss2],feed_dict={gen_input:gen_input_batch,gen_output:gen_output_batch,dLR:d_lr[epoch],gLR:g_lr[epoch]})
            #output information of training process per 100 times
            if mbIdx%100==0 :           
                print("real_logits")
                print(real_logits)
                print("fake_logits1")
                print(fake_logits1)
                print("fake_logits2")
                print(fake_logits2)
                print("**********discriminator_loss")
                print("discriminator_loss=%f"%res_disc_loss)
                print("**********generator1_loss")
                print("gen_loss1:%f=100*%f+10x%f+1*%f(gen_loss1=100*MF+10*cGAN+1*GC)" %(res_gen_loss1,res_MF_loss1,res_adv_loss1,res_GC_loss))
                print("MF_loss1:%f=%f*%f+%f(MF1=lambda1*MD1+FA1)" %(res_MF_loss1,lambda1,res_MD1,res_FA1))
                print("gen_adver_loss1:%f" % res_adv_loss1) 
                print("**********generator2_loss")
                print("gen_loss2:%f=100*%f+10x%f+1*%f(gen_loss2=100*MF+10*cGAN+1*GC)"%(res_gen_loss2,res_MF_loss2,res_adv_loss2,res_GC_loss))
                print("MF_loss2:%f=%f+%fx%f(MF2=MD2+lambda2*FA2)"%(res_MF_loss2,res_MD2,lambda2,res_FA2))
                print("gen_adver_loss2:%f" % res_adv_loss2) 
                print("**********GC_loss")
                print(res_GC_loss) 
                #================================================Performance on training set============================================================
                sum_train_loss = 0 
                sum_train_false_ratio = 0 
                sum_train_detect_ratio = 0
                sum_train_F1 = 0
                for ind in range(len(train_input_names)):
                   if NO_USE_NORMALIZATION: 
                      input_image=np.expand_dims(np.float32(cv2.imread(train_input_names[ind],-1)),axis=0)/255.0
                   else :
                      input_image=np.expand_dims(np.float32(cv2.imread(train_input_names[ind],-1)),axis=0)/255.0
                      input_image[0,:,:,2]=input_image[0,:,:,2]*2-1
                   #channel replace
                   input_image[:,:,:,0] = input_image[:,:,:,2]
                   input_image[:,:,:,1] = input_image[:,:,:,2]                 
                   gt_image=np.expand_dims(np.expand_dims(np.float32(cv2.imread(train_output_names[ind],-1)),axis=0),axis=3)/255.0
                   output_image=sess.run(gen_result1,feed_dict={gen_input:np.expand_dims(input_image[:,:,:,2],axis=3)})
                   output_image=np.minimum(np.maximum(output_image,0.0),1.0)
                   train_loss = np.mean(np.square(output_image-gt_image))
                   sum_train_loss += train_loss
                   train_false_ratio = np.mean(np.maximum(0,output_image-gt_image))
                   sum_train_false_ratio += train_false_ratio
                   train_detect_ratio = np.sum(output_image*gt_image)/np.maximum(np.sum(gt_image),1)
                   sum_train_detect_ratio += train_detect_ratio
                   train_F1 = calculateF1Measure(output_image,gt_image,0.5)
                   sum_train_F1 += train_F1
                   if NO_USE_NORMALIZATION==0:
                      input_image[0,:,:,2] = (input_image[0,:,:,2]+1)/2
                      input_image[:,:,:,0] = input_image[0,:,:,2]
                      input_image[:,:,:,1] = input_image[0,:,:,2] 
                   input_image  = np.squeeze(input_image*255.0)
                   output_image = np.squeeze(output_image*255.0/np.maximum(output_image.max(),0.0001))
                avg_train_loss = sum_train_loss/len(train_input_names)
                avg_train_false_ratio = sum_train_false_ratio/len(train_input_names)
                avg_train_detect_ratio = sum_train_detect_ratio/len(train_input_names)
                avg_train_F1 = sum_train_F1/len(train_input_names)
                print("================train_L2_loss is %f"% (avg_train_loss))               
                print("================falseAlarm_rate is %f"% (avg_train_false_ratio))
                print("================detection_rate is %f"% (avg_train_detect_ratio))
                print("================F1 measure is %f"% (avg_train_F1))
                #================================================Performance on validation  set============================================================
                sum_val_loss = 0
                sum_val_false_ratio = 0 
                sum_val_detect_ratio = 0
                sumRealTarN = 0
                sumDetTarN = 0
                sum_val_F1 = 0
                for ind in range(len(val_names)):
                   if NO_USE_NORMALIZATION: 
                      input_image=np.expand_dims(np.float32(cv2.imread(val_names[ind],-1)),axis=0)/255.0
                   else :
                      input_image=np.expand_dims(np.float32(cv2.imread(val_names[ind],-1)),axis=0)/255.0
                      input_image[0,:,:,2]=input_image[0,:,:,2]*2-1
                   #channel replace
                   input_image[:,:,:,0] = input_image[:,:,:,2]
                   input_image[:,:,:,1] = input_image[:,:,:,2]
                   gt_image=np.expand_dims(np.expand_dims(np.float32(cv2.imread(val_gt_names[ind],-1)),axis=0),axis=3)/255.0
                   output_image1=sess.run(gen_result1,feed_dict={gen_input:np.expand_dims(input_image[:,:,:,2],axis=3)})
                   output_image1=np.minimum(np.maximum(output_image1,0.0),1.0)
 
                   output_image2=sess.run(gen_result2,feed_dict={gen_input:np.expand_dims(input_image[:,:,:,2],axis=3)})
                   output_image2=np.minimum(np.maximum(output_image2,0.0),1.0)
 
                   output_image3=sess.run(fusion_result,feed_dict={gen_input:np.expand_dims(input_image[:,:,:,2],axis=3)})
                   output_image3=np.minimum(np.maximum(output_image3,0.0),1.0)
 
                   output_image = output_image2
                   val_loss = np.mean(np.square(output_image-gt_image))
                   sum_val_loss += val_loss
                   val_false_ratio = np.mean(np.maximum(0,output_image-gt_image))
                   sum_val_false_ratio += val_false_ratio
                   val_detect_ratio = np.sum(output_image*gt_image)/np.maximum(np.sum(gt_image),1)
                   sum_val_detect_ratio += val_detect_ratio
                   val_F1 = calculateF1Measure(output_image,gt_image,0.5)
                   sum_val_F1 += val_F1
                   if NO_USE_NORMALIZATION==0:
                      input_image[0,:,:,2] = (input_image[0,:,:,2]+1)/2
                      input_image[:,:,:,0] = input_image[0,:,:,2]
                      input_image[:,:,:,1] = input_image[0,:,:,2] 
                   input_image  = np.squeeze(input_image*255.0)
                   output_image1 = np.squeeze(output_image1*255.0)
                   output_image2 = np.squeeze(output_image2*255.0)
                   output_image3 = np.squeeze(output_image3*255.0)
                   cv2.imwrite("%s/%05d_G1.png"%(task,ind),np.uint8(output_image1))
                   cv2.imwrite("%s/%05d_G2.png"%(task,ind),np.uint8(output_image2))
                   cv2.imwrite("%s/%05d_Res.png"%(task,ind),np.uint8(output_image3))
                avg_val_loss = sum_val_loss/len(val_names)
                avg_val_false_ratio  = sum_val_false_ratio/len(val_names)
                avg_val_detect_ratio = sum_val_detect_ratio/len(val_names)
                avg_val_F1 = sum_val_F1/len(val_names)
                print("=========================================================%s"%(task))
                print("================val_L2_loss is %f"% (avg_val_loss))
                print("================falseAlarm_rate is %f"% (avg_val_false_ratio))
                print("================detection_rate is %f"% (avg_val_detect_ratio))
                print("================F1 measure is %f"% (avg_val_F1))
                print("saved at epoch=%05d,mb=%05d(%05d)"%(epoch,mbIdx,miniBatchNum))
                saver.save(sess,"%s/model.ckpt"%task)
          
 
test_names = val_names
print(len(test_names))

block_reconst = False  #Extract overlapping patches and put them into our model, and then build the whole segmentation result by using the segmentation results of those patches.

for ind in range(len(test_names)):
    if not os.path.isfile(test_names[ind]):
        continue
    if NO_USE_NORMALIZATION: 
        input_image=np.expand_dims(np.float32(cv2.imread(test_names[ind],-1)),axis=0)/255.0
    else :
        input_image=np.expand_dims(np.float32(cv2.imread(test_names[ind],-1)),axis=0)/255.0
        input_image[0,:,:,2]=input_image[0,:,:,2]*2-1
    #channel replace
    SNR_image = input_image
    input_image[:,:,:,0] = input_image[:,:,:,2] 
    input_image[:,:,:,1] = input_image[:,:,:,2] 
    st=time.time()
    if block_reconst==False :
          output_image = sess.run(gen_result1,feed_dict={gen_input:np.expand_dims(input_image[:,:,:,2],axis=3)})
    else :
          h = input_image.shape[1]
          w = input_image.shape[2]
          input_patch_image = Image2Patch(np.expand_dims(input_image[:,:,:,2],axis=3),128,h,w,20)
          num = input_patch_image.shape[0]
          if num<=mini_batch_size :
             feed_dict={gen_input: input_patch_image}
             output_patch_image = sess.run(fusion_result,feed_dict)
          else :
             output_patch_image = np.zeros((num,128,128,1),dtype="float32")
             for cc in range(0,num,mini_batch_size):
                 proc_num = min(num-cc,mini_batch_size) 
                 feed_dict={gen_input: input_patch_image[cc:cc+proc_num,:,:,:]} 
                 part_output_patch_image = sess.run(fusion_result,feed_dict)
                 output_patch_image[cc:(cc+proc_num),:,:,:] = part_output_patch_image
          output_image = Patch2Image(output_patch_image,128,h,w,20)
    print("time2=%.3f"%(time.time()-st))
    output_image=np.minimum(np.maximum(output_image,0.0),1.0)
    #output_image = (output_image*255.0)/(output_image.max()+0.01)
    output_image = (output_image*255.0)
    #cv2.imwrite("../STD_result/%05d_org.png"%(ind),np.uint8(255.0*np.squeeze(input_image[0,:,:,:])))
    cv2.imwrite("../Rebuttal_result/%05d_res.png"%(ind),np.uint8(np.squeeze(output_image[0,:,:,:])))


