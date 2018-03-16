import numpy as np
import scipy as sp
import caffe
import sys

caffe.set_mode_gpu()
caffe.set_device(0)
alex_root = 'D:/compressing/'
snapshot=alex_root+'UDF002/'
net_name=alex_root+'bvlc_reference_caffenet_jpg.caffemodel'


solver = caffe.SGDSolver(alex_root+'solver_caffe.prototxt')
solver.net.copy_from(net_name)


layer_conv={0:'conv1',1:'conv2',2:'conv3',3:'conv4',4:'conv5'}
layer_ip={0:'fc6',1:'fc7',2:'fc8'}
layer_all={0:'conv1',1:'conv2',2:'conv3',3:'conv4',4:'conv5',5:'fc6',6:'fc7',7:'fc8'}

display=1
test_interval = 20
test_iter=1000
merge_interval=40
train_interval=50
other_train_iter=0
max_distance=1e15
min_distance=-1e15

order=2
mymetric='euclidean'


def count_mask(it):
    total_size_w=0.0
    pruning_size_w=0.0
    for i in range(len(mask[0])):
        total_size_w+=mask[0][i].size
        pruning_size_w+=mask[0][i][mask[0][i]].size

    total_size_b=0.0
    pruning_size_b=0.0
    for i in range(len(mask[1])):
        total_size_b+=mask[1][i].size
        pruning_size_b+=mask[1][i][mask[1][i]].size

    print('Iteration', it, 'Bias Pruning...',pruning_size_b/total_size_b,'Weight Pruning...',pruning_size_w/total_size_w)
    return(pruning_size_w/total_size_w,pruning_size_b/total_size_b)


def merge_weight_eachlayer_i_j(neuron_merge_count_eachlayer,neuron_ifnotmerge_eachlayer,layer_id,i,j):
    layer=layer_all[layer_id]
    solver.net.params[layer][0].data[i]=(solver.net.params[layer][0].data[i]*neuron_merge_count_eachlayer[i]+
        solver.net.params[layer][0].data[j]*neuron_merge_count_eachlayer[j]) / (neuron_merge_count_eachlayer[i]+
        neuron_merge_count_eachlayer[j])
    mask[0][layer_id][j]=True
    solver.net.params[layer][1].data[i]=max(solver.net.params[layer][1].data[i],solver.net.params[layer][1].data[j])
    mask[1][layer_id][j]=True
    
    layer_id+=1
    layer=layer_all[layer_id]
    solver.net.params[layer][0].data[:,i]=solver.net.params[layer][0].data[:,i]+solver.net.params[layer][0].data[:,j]
    mask[0][layer_id][:,j]=True
    
    neuron_ifnotmerge_eachlayer[j]=False
    neuron_merge_count_eachlayer[i]+=neuron_merge_count_eachlayer[j]
    


def get_distance_row(norm,data_w,data_b,neuron_ifnotmerge_eachlayer,i):
    size=data_b.size
    dis_i=np.empty((size))
    dis_i.fill(max_distance)
    for j in range(i+1,size):
        if neuron_ifnotmerge_eachlayer[j]:
            dis_i[j]=np.linalg.norm(x=(data_w[i]-data_w[j]),ord=order)
    
    return dis_i[i:size]

def get_distance_col(norm,data_w,data_b,neuron_ifnotmerge_eachlayer,j):
    size=data_b.size
    dis_j=np.empty((size))
    dis_j.fill(max_distance)
    for i in range(j):
        if neuron_ifnotmerge_eachlayer[i]:
            dis_j[i]=np.linalg.norm(x=(data_w[i]-data_w[j]),ord=order)
    
    return dis_j[:j]

def merge_weight_eachlayer(layer_id):
    data_w=solver.net.params[layer_all[layer_id]][0].data
    data_b=solver.net.params[layer_all[layer_id]][1].data
    data_w_next=solver.net.params[layer_all[layer_id+1]][0].data
    data_b_next=solver.net.params[layer_all[layer_id+1]][1].data
    size=data_b.size
    size_next=data_b_next.size
    oncecount=0
            
    norm=np.empty(size)
    for i in range(size):
        norm[i]=np.linalg.norm(data_w[i],ord=order)
        data_w[i]/=norm[i]
        data_b[i]/=norm[i]
        data_w_next[:,i]*=norm[i]

    avg_a_square=np.empty(size)
    for i in range(size):
        avg_a_square[i]=np.power(np.linalg.norm(data_w_next[:,i],ord=order),2)

        
    dis=np.empty((size,size))
    dis.fill(max_distance)
    for i in range(size):
        if neuron_ifnotmerge[layer_id][i]:
            dis[i,i:size]=get_distance_row(norm,data_w,data_b,neuron_ifnotmerge[layer_id],i)
            for j in range(size):
                if neuron_ifnotmerge[layer_id][j]:
                    dis[i,j]=dis[i,j]*avg_a_square[j]
                    
    while oncecount<neuron_merge_count_eachtime[layer_id]:
        i=dis.argmin()//size
        j=dis.argmin()%size
        print(i,j, dis.min(), norm[i], norm[j])
        merge_weight_eachlayer_i_j(neuron_merge_count[layer_id],neuron_ifnotmerge[layer_id],layer_id,i,j)

        norm_buff=np.linalg.norm(data_w[i],ord=order)
        data_w[i]/=norm_buff
        data_b[i]/=norm_buff
        data_w_next[:,i]*=norm_buff
        norm[i]=norm_buff
        
        avg_a_square[i]=np.power(np.linalg.norm(data_w_next[:,i],ord=order),2)
        dis[i,i:size]=get_distance_row(norm,data_w,data_b,neuron_ifnotmerge[layer_id],i)
        dis[0:i,i]=get_distance_col(norm,data_w,data_b,neuron_ifnotmerge[layer_id],i)

        for k in range(size):
            if neuron_ifnotmerge[layer_id][k]:
                dis[i,k]=avg_a_square[k]*dis[i,k]
        for k in range(size):
            if neuron_ifnotmerge[layer_id][k]:
                dis[k,i]=avg_a_square[i]*dis[k,i]
        
        for k in range(size):
            dis[j,k]=max_distance
        for k in range(size):
            dis[k,j]=max_distance
        oncecount+=1
    
def run_solver_pruning(niter):
    train_loss = np.zeros((niter+other_train_iter)//display+1)
    test_acc = np.zeros((niter+other_train_iter)//test_interval+1 )
    pruning_rate_w = np.zeros((niter+other_train_iter)//display+1)
    pruning_rate_b = np.zeros((niter+other_train_iter)//display+1)
    layer_id=len(layer_all)-1
    time=0
    for it in range(niter):
        sys.stdout.flush()
        if it % merge_interval==0:
            while time==neuron_merge_time[layer_id]:
                layer_id-=1
                if layer_id<0:
                    break
                time=0
            if layer_id>=0:
                merge_weight_eachlayer(layer_id)
                for i in range(len(layer_all)):
                    solver.net.params[layer_all[i]][0].data[mask[0][i]]=0
                    solver.net.params[layer_all[i]][1].data[mask[1][i]]=0
                correct = 0
                for test_it in range(test_iter):
                    solver.test_nets[0].forward()
                    correct += solver.test_nets[0].blobs['accuracy'].data
                test_acc[it // test_interval] = correct / test_iter
                print('Iteration', it, 'testing...','acc...',test_acc[it // test_interval])
                time+=1

               
        for i in range(len(layer_all)):
            solver.net.params[layer_all[i]][0].data[mask[0][i]]=0
            solver.net.params[layer_all[i]][1].data[mask[1][i]]=0
        solver.step(train_interval)
        for i in range(len(layer_all)):
            solver.net.params[layer_all[i]][0].data[mask[0][i]]=0
            solver.net.params[layer_all[i]][1].data[mask[1][i]]=0
        if it % display ==display-1:
            train_loss[it//display] = solver.net.blobs['loss'].data
            print('Iteration', it, 'training...','loss...',train_loss[it//display])
            (pruning_rate_w[it//display],pruning_rate_b[it//display])=count_mask(it)

        if it % test_interval == test_interval-1:
            correct = 0
            for test_it in range(test_iter):
                solver.test_nets[0].forward()
                correct += solver.test_nets[0].blobs['accuracy'].data
            test_acc[it // test_interval] = correct / test_iter
            print('Iteration', it, 'testing...','acc...',test_acc[it // test_interval])
        
    for it in range(other_train_iter):
        
        for i in range(len(layer_all)):
            solver.net.params[layer_all[i]][0].data[mask[0][i]]=0
            solver.net.params[layer_all[i]][1].data[mask[1][i]]=0
        solver.step(train_interval)
        for i in range(len(layer_all)):
            solver.net.params[layer_all[i]][0].data[mask[0][i]]=0
            solver.net.params[layer_all[i]][1].data[mask[1][i]]=0
            
        if it % display ==display-1:
            train_loss[(it+niter)//display] = solver.net.blobs['loss'].data
            print('Other Iteration', it, 'training...','loss...',train_loss[(it+niter)//display])

        if it % test_interval == test_interval-1:
            correct = 0
            for test_it in range(test_iter):
                solver.test_nets[0].forward()
                correct += solver.test_nets[0].blobs['accuracy'].data
            test_acc[(it+niter) // test_interval] = correct / test_iter
            print('Other Iteration', it, 'testing...','acc...',test_acc[(it+niter) // test_interval])
    
def init(solver):
    niter = 0
    for i in range(len(layer_all)):
        niter+=neuron_merge_time[i]*merge_interval
    niter+=1
    mask=[]
    mask.append([])
    mask.append([])
    for i in range(len(layer_all)):
        mask[0].append(np.zeros_like(solver.net.params[layer_all[i]][0].data,dtype='bool'))
        mask[1].append(np.zeros_like(solver.net.params[layer_all[i]][1].data,dtype='bool'))
    
    neuron_ifnotmerge=[]
    for i in range(len(layer_all)):
        neuron_ifnotmerge.append(np.ones(len(solver.net.params[layer_all[i]][1].data),dtype='bool'))
    
    neuron_merge_count=[]
    for i in range(len(layer_all)):
        neuron_merge_count.append(np.ones(len(solver.net.params[layer_all[i]][1].data),dtype='int32'))

    solver.net.copy_from(net_name)
    solver.step(1)
    return niter,mask,neuron_ifnotmerge,neuron_merge_count


neuron_merge_time=[0,0,0,0,0,0,32,0]
neuron_merge_count_eachtime=[0,0,0,0,0,4096,128,1]
niter,mask,neuron_ifnotmerge,neuron_merge_count=init(solver)    
run_solver_pruning(niter)

neuron_merge_time=[0,0,0,0,0,32,0,0]
neuron_merge_count_eachtime=[0,0,0,0,0,128,128,1]
niter,mask,neuron_ifnotmerge,neuron_merge_count=init(solver)    
run_solver_pruning(niter)


