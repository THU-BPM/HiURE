import argparse
import builtins
import math
import random
import shutil
import warnings
from tqdm import tqdm
import numpy as np
import faiss

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

import models.model_builder
from models import bert_model
from transformers import AdamW
from transformers import BertTokenizer
import torch
from torch import nn
import os
import time, json
from torch.utils.data import TensorDataset
import scorer
import traceback

parser = argparse.ArgumentParser(description='PyTorch PCL for relation extraction Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=3, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=35, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--low-dim', default=768*3, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--models-r', default=10, type=int,  # 10  640  16384
                    help='queue size; number of negative pairs; needs to be smaller than num_cluster (default: 16384)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--temperature', default=0.02, type=float,
                    help='softmax temperature')

parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco-v2/SimCLR data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

# 10,13,16,20,50,80,100,250,500,1000,1500,3000,6000,10000,20000
parser.add_argument('--num-cluster', default='10,16,100', type=str,  # 10,16,100  800,1000,1200  25000,50000,100000  10,13,16,20,50,80,100,250,500,1000  ,200,400,600,800,1000,1500,2000,3000,1000
                    help='number of clusters')
parser.add_argument('--warmup-epoch', default=0, type=int,
                    help='number of warm-up epochs to only train with InfoNCE loss')
parser.add_argument('--exp-dir', default='10clusters_train', type=str,
                    help='experiment directory')

parser.add_argument('--max-length', default=128, type=int,
                    help='max length of sentence to be feed into bert (default 128)')
parser.add_argument('--eps', default=1e-8, type=float,
                    help='adam_epsilon (default: 1e-8)')
parser.add_argument('--add-word-num', default=3, type=int,
                    help='data augmentation words number (default 3)')
parser.add_argument('--repeat-index', default=0, type=int,
                    help='repeat run index')

# ------------------------init parameters----------------------------

CUDA = "0,1,2,3,4,5,6"
DATASET = 'tacred_no_relation/'
add_word_num = 3
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA
checkpoint_dir = '/home/xuminghu/ACL2021/source_code/train_result_warm2'
test_mode = False


def main():
    args = parser.parse_args()

    if test_mode:
        args.warmup_epoch = 3
        DATASET = 'tacred/tacred'
        args.resume = ''

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    args.num_cluster = args.num_cluster.split(',')

    args.low_dim = (args.add_word_num + 2) * 768

    if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        print(args.dist_url)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    bert_encoder = bert_model.RelationClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=args.low_dim,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    model = models.model_builder.MoCo(
        bert_encoder,
        args.low_dim, args.pcl_r, args.moco_m, args.temperature, args.mlp)
    # print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = AdamW(model.parameters(),
                      lr=args.lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=args.eps  # args.adam_epsilon  - default is 1e-8.
                      )

    cudnn.benchmark = True

    sentence_train = json.load(open('/home/xuminghu/ACL2021/data/' + DATASET + 'test_sentence.json', 'r'))
    sentence_train_label = json.load(open('/home/xuminghu/ACL2021/data/' + DATASET + 'test_label_id.json', 'r'))
    train_dataset, eval_dataset = pre_processing(sentence_train, sentence_train_label, args)

    # sentence_val = json.load(open('/home/xuminghu/ACL2021/data/' + DATASET + '/test_sentence.json', 'r'))
    # sentence_val_label = json.load(open('/home/xuminghu/ACL2021/data/' + DATASET + '/test_label_id.json', 'r'))
    # val_dataset = pre_processing(sentence_val, sentence_val_label, is_eval=True)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False)
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=False)
    else:
        train_sampler = None
        eval_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, sampler=train_sampler, drop_last=True)

    # dataloader for center-cropped images, use larger batch size to increase speed
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        sampler=eval_sampler, num_workers=args.workers, drop_last=True)

    gold = []
    for eval_data in eval_dataset:
        gold.append(eval_data[2].item())
    # print(gold)

    # train_dataloader = labeled_dataloader
    # cur_dir = os.listdir()
    cur_dir = ['train_15clusters_3add_words_10epochs']
    for check_dir in cur_dir:
        if os.path.isdir(check_dir) and 'measurements.txt' in os.listdir(check_dir):

            checkpoints=os.listdir(check_dir)
            checkpoints.sort(reverse=True)

            measure_file = open(os.path.join(check_dir, 'evaluate.txt'), 'w')
            # measure_file.write('Args: {}\n'.format(args))

            for item in checkpoints:
                if '.pth.tar' not in item:
                    continue
                chpt = os.path.join(check_dir,item)
                if os.path.isfile(chpt):
                    print("=> loading checkpoint '{}'".format(item))
                    try:
                        if args.gpu is None:
                            checkpoint = torch.load(chpt)
                        else:
                            # Map model to be loaded to specified single gpu.
                            loc = 'cuda:{}'.format(args.gpu)
                            checkpoint = torch.load(chpt, map_location=loc)
                        args.start_epoch = checkpoint['epoch']
                        model.load_state_dict(checkpoint['state_dict'])
                        optimizer.load_state_dict(checkpoint['optimizer'])
                        print("=> loaded checkpoint '{}' (epoch {})"
                              .format(item, checkpoint['epoch']))

                        measure_file.write('checkpoint: {}\n'.format(item))
                        features = compute_features(eval_loader, model, args)
                        # placeholder for clustering result
                        cluster_result = {'relation2cluster': [], 'centroids': [], 'density': []}
                        for num_cluster in args.num_cluster:
                            cluster_result['relation2cluster'].append(
                                torch.zeros(len(eval_dataset), dtype=torch.long).cuda())
                            cluster_result['centroids'].append(torch.zeros(int(num_cluster), args.low_dim).cuda())
                            cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda())

                        if args.gpu == 0:
                            features[
                                torch.norm(features,
                                           dim=1) > 1.5] /= 2  # account for the few samples that are computed twice
                            features = features.numpy()
                            cluster_result = run_kmeans(features, args)  # run kmeans clustering on master node
                            calculate_measurements(cluster_result, args, gold, measure_file)
                    except Exception as e:
                        print(e)
                        traceback.print_exc()
                        measure_file.write('checkpoint not satisfied data! Exception：{}\n'.format(e))
                        break
                else:
                    print("=> no checkpoint found at '{}'".format(chpt))



            measure_file.close()


def evaluate(train_loader, model, criterion, optimizer, args, chpt,cluster_result=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc_inst = AverageMeter('Acc@Inst', ':6.2f')
    acc_proto = AverageMeter('Acc@Proto', ':6.2f')

    # precision_meter = AverageMeter('Precision', ':6.2f')
    # recall_meter =  AverageMeter('Recall', ':6.2f')
    # fscore_meter =  AverageMeter('Fscore', ':6.2f')
    # homo_meter = AverageMeter('Homo', ':6.2f')
    # v_m_meter = AverageMeter('V_m', ':6.2f')
    # ari_meter = AverageMeter('ARI', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc_inst, acc_proto,], #precision_meter,recall_meter,fscore_meter,homo_meter,v_m_meter,ari_meter],
        prefix="Checkpoint: [{}]".format(chpt))

    # switch to train mode
    model.train()

    end = time.time()
    model_result = []
    for num_cluster_index in range(len(args.num_cluster)):
        model_result.append([])
    for i, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # batch.requires_grad=True
        if args.gpu is not None:
            # batch[0] = batch[0].cuda(args.gpu, non_blocking=True)
            # batch[1] = batch[1].cuda(args.gpu, non_blocking=True)
            for b_i in range(len(batch)):
                batch[b_i] = batch[b_i].cuda(args.gpu, non_blocking=True)

        # compute output
        output, target, output_proto, target_proto = model(sen_q=batch[:6], sen_k=batch[:5] + [batch[6]],
                                                           cluster_result=cluster_result, index=batch[-1])
        for num_cluster_index in range(len(args.num_cluster)):
            model_result[num_cluster_index].extend(target_proto[num_cluster_index].cpu().detach().numpy())
        # for tp in target_proto:
        #     model_result+=tp.cpu().detach().numpy()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    print(model_result)


def compute_features(eval_loader, model, args):
    print('Computing features...')
    model.eval()
    features = torch.zeros(len(eval_loader.dataset), args.low_dim).cuda()
    for i, sentence in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            # sentence = sentence.cuda(non_blocking=True)
            for j in range(len(sentence)):
                sentence[j] = sentence[j].cuda(non_blocking=True)
            feat = model(sen_q=sentence, is_eval=True)
            features[sentence[-1]] = feat  # .view(args.low_dim*args.batch_size)
    dist.barrier()
    dist.all_reduce(features, op=dist.ReduceOp.SUM)
    return features.cpu()


# ------------------------functions----------------------------

def calculate_measurements(cluster_result,args,gold,measure_file):
    def print_output(content):
        measure_file.write(content)
        measure_file.write('\n')
        print(content)
        pass
    count = 0
    for cluster_instance in cluster_result['relation2cluster']:
        print_output('Cluster num: {}'.format(args.num_cluster[count]))
        count += 1
        pred = cluster_instance.cpu().numpy()
        print(pred)
        p, r, f1 = scorer.bcubed_score(gold, pred)
        print_output('B cube: p={:.5f} r={:.5f} f1={:.5f}'.format(p, r, f1))

        ari = scorer.adjusted_rand_score(gold, pred)
        homo, comp, v_m = scorer.v_measure(gold, pred)
        print_output('V-measure: hom.={:.5f} com.={:.5f} vm.={:.5f}'.format(homo, comp, v_m))
        print_output('ARI={:.5f}'.format(ari))
        print_output('')

    measure_file.write('\n')
    pass

def random_select_close_words(target_index, start, end, add_num):
    selected_index = [target_index]
    close_degree = 4
    while close_degree > 0 and len(selected_index) <= add_num:
        selected_range = [i for i in range(int(target_index - (target_index - start) / close_degree),
                                           int(target_index + (end - target_index) / close_degree)) if
                          i not in selected_index]
        selected_index = selected_index + selected_range[:add_num]
        close_degree -= 1
    if len(selected_index) <= add_num:
        selected_index += [target_index] * (add_num - len(selected_index))
    return selected_index[1:target_index + 1]
    pass


def word_level_augmentation(pos1, pos2, validate_length):
    if pos1 > pos2:
        tmp = pos2
        pos2 = pos1
        pos1 = tmp
    add_num = add_word_num
    middle_range = [i for i in range(pos1 + 1, pos2)]
    random.shuffle(middle_range)
    selected_index = middle_range[:add_num]
    if len(selected_index) < add_num:
        add_num = add_num - len(selected_index)

        one_num = int(add_num / 2)
        selected_index += random_select_close_words(pos1, 0, pos1, one_num)
        selected_index += random_select_close_words(pos2, pos2, validate_length, add_num - one_num)
    return selected_index
    pass


# ------------------------prepare sentences----------------------------

# Tokenize all of the sentences and map the tokens to thier word IDs.
def pre_processing(sentence_train, sentence_train_label, args):
    input_ids = []
    attention_masks = []
    labels = []
    e1_pos = []
    e2_pos = []
    aug_pos_arr1 = []
    aug_pos_arr2 = []
    aug_pos_arr3 = []
    index_arr = []
    # Load tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    counter = 0
    # pre-processing sentenses to BERT pattern
    for i in range(len(sentence_train)):

        encoded_dict = tokenizer.encode_plus(
            sentence_train[i],  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=args.max_length,  # Pad & truncate all sentences.
            truncation=True,  # explicitely truncate examples to max length
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        try:
            # Find e1(id:2487) and e2(id:2475) position
            pos1 = (encoded_dict['input_ids'] == 2487).nonzero()[0][1].item()
            pos2 = (encoded_dict['input_ids'] == 2475).nonzero()[0][1].item()
            e1_pos.append(pos1)
            e2_pos.append(pos2)

            # data augmentation
            aug_pos = word_level_augmentation(pos1, pos2, encoded_dict['input_ids'].nonzero().shape[0])
            aug_pos_arr1.append(aug_pos)
            aug_pos = word_level_augmentation(pos1, pos2, encoded_dict['input_ids'].nonzero().shape[0])
            aug_pos_arr2.append(aug_pos)
            aug_pos = word_level_augmentation(pos1, pos2, encoded_dict['input_ids'].nonzero().shape[0])
            aug_pos_arr3.append(aug_pos)

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])
            labels.append(sentence_train_label[i])
            index_arr.append(counter)

            counter += 1
            # for testing
            if counter >= 1000 and test_mode:
                break
        except:
            pass
            # print(sent)

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    e1_pos = torch.tensor(e1_pos)
    e2_pos = torch.tensor(e2_pos)
    aug_pos_arr1 = torch.tensor(aug_pos_arr1)
    aug_pos_arr2 = torch.tensor(aug_pos_arr2)
    aug_pos_arr3 = torch.tensor(aug_pos_arr3)
    index_arr = torch.tensor(index_arr)

    eval_dataset = TensorDataset(input_ids, attention_masks, labels, e1_pos, e2_pos, aug_pos_arr3, index_arr)

    # Combine the training inputs into a TensorDataset.
    train_dataset = TensorDataset(input_ids, attention_masks, labels, e1_pos, e2_pos, aug_pos_arr1, aug_pos_arr2,
                                  index_arr)
    return train_dataset, eval_dataset


def run_kmeans(x, args):
    """
    Args:
        x: data to be clustered
    """

    print('performing kmeans clustering')
    results = {'relation2cluster': [], 'centroids': [], 'density': []}

    for seed, num_cluster in enumerate(args.num_cluster):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = args.gpu
        index = faiss.GpuIndexFlatL2(res, d, cfg)

        clus.train(x, index)

        D, I = index.search(x, 1)  # for each sample, find cluster distance and assignments
        relation2cluster = [int(n[0]) for n in I]

        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

        # sample-to-centroid distances for each cluster
        Dcluster = [[] for c in range(k)]
        for im, i in enumerate(relation2cluster):
            Dcluster[i].append(D[im][0])

        # concentration estimation (phi)
        density = np.zeros(k)
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                density[i] = d

                # if cluster only has one point, use the max to estimate its concentration
        dmax = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = dmax

        density = density.clip(np.percentile(density, 10),
                               np.percentile(density, 90))  # clamp extreme values for stability
        density = args.temperature * density / density.mean()  # scale the mean to temperature

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)

        relation2cluster = torch.LongTensor(relation2cluster).cuda()
        density = torch.Tensor(density).cuda()

        results['centroids'].append(centroids)
        results['density'].append(density)
        results['relation2cluster'].append(relation2cluster)

    return results


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    # Load models
    main()
