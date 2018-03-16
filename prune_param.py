import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import caffe

caffe.set_mode_gpu()
caffe.set_device(1)
alex_root = 'D:/compressing/'
snapshot=alex_root+'UDF003/'
solver = caffe.SGDSolver(alex_root+'solver_caffe2.prototxt')
#solver.net.copy_from(snapshot+'UDF003.caffemodel')
solver.net.copy_from(alex_root+'_iter_200000.caffemodel')
layer_conv={0:'conv1',1:'conv2',2:'conv3',3:'conv4',4:'conv5'}
layer_ip={5:'fc6',6:'fc7',7:'fc8'}
layer_all=dict(layer_conv,**layer_ip)

exp=1e10
niter =35
display=1
test_interval = 1
test_iter=1000
thershold_base=np.array([0.,0,0,0,0,0.0083,0.0099,0.0113])
thershold_step=np.array([0.,0,0,0,0,0.000077,0.000103,0.000114])
thershold_base=thershold_step*65
train_interval =1000
ip_all_rate=1
step_conv=1
step_ip=1

def count_mask(mask,it):
    total_size_w=0.0
    pruning_size_w=0.0
    for i in range(len(mask[0])):
        total_size_w+=mask[0][i].size
        pruning_size_w+=mask[0][i][mask[0][i]].size
        print layer_all[i],float(mask[0][i][mask[0][i]].size)/float(mask[0][i].size)

    total_size_b=0.0
    pruning_size_b=0.0
    for i in range(len(mask[1])):
        total_size_b+=mask[1][i].size
        pruning_size_b+=mask[1][i][mask[1][i]].size

    print('Iteration...', it, 'Bias Pruning...',pruning_size_b/total_size_b,'Weight Pruning...',pruning_size_w/total_size_w)
    return(pruning_size_w/total_size_w,pruning_size_b/total_size_b)

def plot_loss_acc_pruningrate(filename,train_loss,test_acc,pruning_rate_w,pruning_rate_b):
    plt.figure()
    ax1=plt.subplot(2,2,1)
    ax2=plt.subplot(2,2,2)
    ax3=plt.subplot(2,2,3)
    ax4=plt.subplot(2,2,4)
    ax1.plot(display*np.arange(len(train_loss)), train_loss)
    ax2.plot(test_interval * np.arange(len(test_acc)), test_acc)
    ax3.plot(display*np.arange(len(pruning_rate_w)), pruning_rate_w)
    ax4.plot(display*np.arange(len(pruning_rate_b)), pruning_rate_b)
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss')
    ax2.set_xlabel('iteration')
    ax2.set_ylabel('test accuracy')
    ax3.set_xlabel('iteration')
    ax3.set_ylabel('pruning rate weight')
    ax4.set_xlabel('iteration')
    ax4.set_ylabel('pruning rate bias')
    plt.savefig(filename)


def run_solver_pruning(solver,niter,mask,thershold_base,test_interval,train_interval):
    train_loss = np.zeros(niter/display+1)
    test_acc = np.zeros(niter/test_interval)
    pruning_rate_w = np.zeros(niter/display+1)
    pruning_rate_b = np.zeros(niter/display+1)
    thershold=thershold_base
    for it in range(int(niter*ip_all_rate)):
        if it % step_ip==0:
            thershold+=thershold_step
        for layer_id in range(len(layer_conv),len(layer_all)):
            mask[0][layer_id][np.abs(solver.net.params[layer_all[layer_id]][0].data)<thershold[layer_id]]=True
        for i in range(len(layer_all)):
            solver.net.params[layer_all[i]][0].data[mask[0][i]]=0
        solver.step(train_interval)
        for i in range(len(layer_all)):
            solver.net.params[layer_all[i]][0].data[mask[0][i]]=0
        if it % display ==0:
            train_loss[it//display] = solver.net.blobs['loss'].data
            print('Iteration', it, 'training...','loss...',train_loss[it//display])
            (pruning_rate_w[it//display],pruning_rate_b[it//display])=count_mask(mask,it)

        if it % test_interval == 0:
            correct = 0
            for test_it in range(test_iter):
                solver.test_nets[0].forward()
                correct += solver.test_nets[0].blobs['accuracy'].data
            test_acc[it // test_interval] = correct / test_iter
            print('Iteration', it, 'testing...','acc...',test_acc[it // test_interval])
    plot_loss_acc_pruningrate(snapshot+'UDF003.jpg',train_loss,test_acc,pruning_rate_w,pruning_rate_b)

mask=[]
mask.append([])
mask.append([])
for i in range(len(layer_all)):
    mask[0].append(np.zeros_like(solver.net.params[layer_all[i]][0].data,dtype='bool'))
    mask[1].append(np.zeros_like(solver.net.params[layer_all[i]][1].data,dtype='bool'))
run_solver_pruning(solver,niter,mask,thershold_base,test_interval,train_interval)
np.save(snapshot+'mask_UDF003.npy',mask)
solver.net.save(snapshot+'UDF003.caffemodel')
