from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score
from sklearn import metrics
import torch
import numpy as np
from scipy.sparse import coo_matrix

# cited: https://github.com/ttthy/ure/blob/a162c7a1613618282d4dd029239028d287733a15/ure/scorer.py
def bcubed_correctness(gold, pred, na_id=-1):
    # remove NA
    gp = [(x,y) for x, y in zip(gold, pred) if x != na_id]
    gold = [x for x,_ in gp]
    pred = [y for _,y in gp]

    # compute 'correctness'
    l = len(pred)
    assert(len(gold) == l)
    gold = torch.IntTensor(gold)
    pred = torch.IntTensor(pred)
    gc = ((gold.unsqueeze(0) - gold.unsqueeze(1)) == 0).int()
    pc = ((pred.unsqueeze(0) - pred.unsqueeze(1)) == 0).int()
    c = gc * pc
    return c, gc, pc


def bcubed_precision(c, gc, pc):
    pcsum = pc.sum(1)
    total = torch.where(pcsum > 0, pcsum.float(), torch.ones(pcsum.shape))
    return ((c.sum(1).float() / total).sum() / gc.shape[0]).item()


def bcubed_recall(c, gc, pc):
    gcsum = gc.sum(1)
    total = torch.where(gcsum > 0, gcsum.float(), torch.ones(gcsum.shape))
    return ((c.sum(1).float() / total).sum() / pc.shape[0]).item()


def bcubed_score(gold, pred, na_id=-1):
    c, gc, pc = bcubed_correctness(gold, pred, na_id)
    prec = bcubed_precision(c, gc, pc)
    rec = bcubed_recall(c, gc, pc)
    return prec, rec, 2 * (prec * rec) / (prec + rec)


def v_measure(gold, pred):
    homo = homogeneity_score(gold, pred)
    comp = completeness_score(gold, pred)
    v_m = v_measure_score(gold, pred)
    return homo, comp, v_m


def check_with_bcubed_lib(gold, pred):
    import bcubed
    ldict = dict([('item{}'.format(i), set([k])) for i, k in enumerate(gold)])
    cdict = dict([('item{}'.format(i), set([k])) for i, k in enumerate(pred)])

    precision = bcubed.precision(cdict, ldict)
    recall = bcubed.recall(cdict, ldict)
    fscore = bcubed.fscore(precision, recall)

    print('P={} R={} F1={}'.format(precision, recall, fscore))

    return precision, recall, fscore

def contingency_matrix(ref_labels, sys_labels):
    """Return contingency matrix between ``ref_labels`` and ``sys_labels``."""
    ref_classes, ref_class_inds = np.unique(ref_labels, return_inverse=True)
    sys_classes, sys_class_inds = np.unique(sys_labels, return_inverse=True)
    n_frames = ref_labels.size
    # Following works because coo_matrix sums duplicate entries. Is roughly
    # twice as fast as np.histogram2d.
    cmatrix = coo_matrix(
        (np.ones(n_frames), (ref_class_inds, sys_class_inds)),
        shape=(ref_classes.size, sys_classes.size),
        dtype=np.int)
    cmatrix = cmatrix.toarray()
    return cmatrix, ref_classes, sys_classes

def my_bcubed(ref_labels, sys_labels, cm=None):
    """Return B-cubed precision, recall, and F1.
    The B-cubed precision of an item is the proportion of items with its
    system label that share its reference label (Bagga and Baldwin, 1998).
    Similarly, the B-cubed recall of an item is the proportion of items
    with its reference label that share its system label. The overall B-cubed
    precision and recall, then, are the means of the precision and recall for
    each item.
    Parameters
    ----------
    ref_labels : ndarray, (n_frames,)
        Reference labels.
    sys_labels : ndarray, (n_frames,)
        System labels.
    cm : ndarray, (n_ref_classes, n_sys_classes)
        Contingency matrix between reference and system labelings. If None,
        will be computed automatically from ``ref_labels`` and ``sys_labels``.
        Otherwise, the given value will be used and ``ref_labels`` and
        ``sys_labels`` ignored.
        (Default: None)
    Returns
    -------
    precision : float
        B-cubed precision.
    recall : float
        B-cubed recall.
    f1 : float
        B-cubed F1.
    References
    ----------
    Bagga, A. and Baldwin, B. (1998). "Algorithms for scoring coreference
    chains." Proceedings of LREC 1998.
    """
    ref_labels = np.array(ref_labels)
    sys_labels = np.array(sys_labels)
    if cm is None:
        cm, _, _ = contingency_matrix(ref_labels, sys_labels)
    cm = cm.astype('float64')
    cm_norm = cm / cm.sum()
    precision = np.sum(cm_norm * (cm / cm.sum(axis=0)))
    recall = np.sum(cm_norm * (cm / np.expand_dims(cm.sum(axis=1), 1)))
    f1 = 2*(precision*recall)/(precision + recall)
    return precision, recall, f1

def calculate_v_m(gold,pred):
    v_homo = metrics.homogeneity_score(gold,pred)
    v_comp = metrics.completeness_score(gold,pred)
    v_measurements = 2*v_comp*v_homo/(v_comp+v_homo)
    return v_homo,v_comp,v_measurements
    pass

if __name__ == '__main__':
    gold = [0, 0, 0, 0, 0, 1, 1, 2, 1, 3, 4, 1, 1, 1]
    pred = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]

    # B3
    print(bcubed_score(gold, pred, na_id = 1000), 'should be 0.69')

    # P, R, F1
    check_with_bcubed_lib(gold, pred)

    # homo, v_m, ar
    homo = homogeneity_score(gold, pred)
    v_m = v_measure_score(gold, pred)
    ari = adjusted_rand_score(gold, pred)
    print(homo, v_m, ari)