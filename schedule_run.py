import os
import sys
import time

# This script runs train process with schedule

def gpu_info(gpu_number=5):
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    gpu_memory = int(gpu_status[2+gpu_number*4].split('/')[0].split('M')[0].strip())
    gpu_power = int(gpu_status[1+gpu_number*4].split('   ')[-1].split('/')[0].split('W')[0].strip())
    return gpu_power, gpu_memory


def narrow_setup(cmd,interval=2):
    gpu_power, gpu_memory = gpu_info()
    i = 0
    while gpu_memory > 100 :  # set waiting condition
        gpu_power, gpu_memory = gpu_info()
        i = i % 5
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        gpu_power_str = 'gpu power:%d W |' % gpu_power
        gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
        sys.stdout.write('\r' + gpu_memory_str + ' ' + gpu_power_str + ' ' + symbol)
        sys.stdout.flush()
        time.sleep(interval)
        i += 1
    print('\n' + cmd)
    os.system(cmd)

def parameter_run(param_list,param_name,param_list2=None,param_name2=None):
    cmd_prefix = 'export CUDA_VISIBLE_DEVICES=1,3,4,6; nohup /home/xuminghu/.conda/envs/nlpenv/bin/python -u main_HiURE.py --mlp --aug-plus --cos --use-relation-span '
    cmd_suffix = ' --dist-url tcp://localhost:'+connection_port+' --multiprocessing-distributed --world-size 1 --rank 0  /home/xuminghu/ACL2021/source_code/NYT-FB_data_process/relation_span_ >> final_nyt_console.log 2>&1 &'
    # t_arr = [i*0.01 for i in range(1,20)]
    for i,p in enumerate(param_list):
        if param_name2:
            for p2 in param_list2:
                contrast_str = ''
                if param_name2 == 'num-cluster' and param_name == 'repeat-index' and p==3:
                    contrast_str = '--models-r '+p2.split(',')[0]
                new_cmd = cmd_prefix + '--' + param_name + ' ' + str(p)+ ' --' + param_name2 + ' ' + str(p2) + contrast_str +cmd_suffix
                # narrow_setup(new_cmd, interval=10)
                narrow_port_run(new_cmd, interval=10)
                time.sleep(30)
        else:
            new_cmd = cmd_prefix + '--'+param_name+' ' + str(p) + cmd_suffix
            if i%2==1:
                new_cmd+='--attention'
            # narrow_setup(new_cmd, interval=10)
            narrow_port_run(new_cmd, interval=10)
            time.sleep(30)


def narrow_port_run(cmd,interval=5):
    while port_is_running():  # set waiting condition
        time.sleep(interval)
    print('\n' + cmd)
    os.system(cmd)

def port_is_running():
    port_status = os.popen('lsof -i:'+connection_port).read()
    return len(port_status)>0


if __name__ == '__main__':
    # make schedule
    t_arr = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.5, 1]
    lr_arr = [0.000001,0.0000025,0.000005,0.0000075,0.00001]
    index_arr = [i for i in range(1,7)]
    temp= [0.02,0.2,0.002]

    num_c = ['10,13,16,20,50,80,100,250,500,1000,1500,3000,6000,10000,20000','10,16,100','16,20,50,80,100,250,500,1000']
    num_cluster = ['2,4,6,8,10,16','3,4,6,10,16','6,10,16,100','10,16,100','10,13,16,20,50,80,100,250,500,1000,1500,3000,6000,10000,20000']
    alpha=[1,5,7,9,10,15,20,50,100,1000]
    add_word_num = [i for i in range(1,5)]

    connection_port = '10012'
    data_path = ['/home/xuminghu/ACL2021/data/tacred/tacred/relation_span/','/home/xuminghu/ACL2021/source_code/NYT-FB_data_process/relation_span_']
    # run with schedule, indicate params
    parameter_run(index_arr,'repeat-index')
