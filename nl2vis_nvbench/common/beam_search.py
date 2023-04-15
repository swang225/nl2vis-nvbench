import torch


def one_step_search(
        model,
        src,
        src_mask,
        sos_id
):

    predicted = [[sos_id]]
    predicted = torch.tensor(predicted)

    model.eval()

    with torch.no_grad():
        output, _ = model(src, src_mask, predicted)

    topk = torch.topk(output, k=1, dim=2, sorted=True)
    topkv = topk.values[:, -1, :].tolist()[0]
    topki = topk.indices[:, -1, :].tolist()[0]
    topk = zip(topki, topkv)

    predicted = torch.tensor([topki])

    return predicted


def beam_search(
        model,
        src,
        src_mask,
        sos_id,
        eos_id,
        predicted=None,
        k=1,
        score=0,
        max_len=3
):

    if predicted is None:
        predicted = [[sos_id]]
        predicted = torch.tensor(predicted)

    if (
            predicted[0][-1].tolist() == eos_id or
            len(predicted[0]) >= max_len + 2
    ):
        return predicted, score / (len(predicted[0]) - 1)

    model.eval()

    with torch.no_grad():
        output, _ = model(src, src_mask, predicted)

    topk = torch.topk(output, k=k, dim=2, sorted=True)
    topkv = topk.values[:, -1, :].tolist()[0]
    topki = topk.indices[:, -1, :].tolist()[0]
    topk = zip(topki, topkv)

    cur_res = None
    cur_avg_score = None
    for i, v in topk:

        new_predicted = torch.tensor([predicted[0].tolist() + [i]])
        new_res, new_avg_score = beam_search(
            model, src, src_mask, sos_id, eos_id, new_predicted, k=k, score=score+v)

        if (
                cur_avg_score is None or
                new_avg_score > cur_avg_score
        ):
            cur_res = new_res
            cur_avg_score = new_avg_score

    return cur_res, cur_avg_score
