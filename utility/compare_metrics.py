
from emitters import Emitter


def validate(pred_set, truth_set):
    # pred_set = Emitter.from_result_tensor(output.cpu().detach().numpy(), 0.3)
    # truth_set = Emitter.from_ground_truth(truth[1].numpy())
    fn = truth_set - pred_set
    fp = pred_set - truth_set
    tp = pred_set % truth_set
    jac = tp.length / (tp.length + fp.length + fn.length)
    return jac