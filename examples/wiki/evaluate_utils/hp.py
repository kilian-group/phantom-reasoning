from .hotpot_evaluate_v1 import update_answer, update_sp


def eval(prediction, gold):
    metrics = {
        "em": 0,
        "f1": 0,
        "prec": 0,
        "recall": 0,
        "sp_em": 0,
        "sp_f1": 0,
        "sp_prec": 0,
        "sp_recall": 0,
        "joint_em": 0,
        "joint_f1": 0,
        "joint_prec": 0,
        "joint_recall": 0,
    }
    for dp in gold:
        cur_id = dp["_id"]
        can_eval_joint = True
        if cur_id not in prediction["answer"]:
            print(f"missing answer {cur_id}")
            can_eval_joint = False
        else:
            em, prec, recall = update_answer(metrics, prediction["answer"][cur_id], dp["answer"])
        if cur_id not in prediction["sp"]:
            print(f"missing sp fact {cur_id}")
            can_eval_joint = False
        else:
            sp_em, sp_prec, sp_recall = update_sp(metrics, prediction["sp"][cur_id], dp["supporting_facts"])

        if can_eval_joint:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.0
            joint_em = em * sp_em

            metrics["joint_em"] += joint_em
            metrics["joint_f1"] += joint_f1
            metrics["joint_prec"] += joint_prec
            metrics["joint_recall"] += joint_recall

    N = len(gold)
    for k in metrics.keys():
        metrics[k] /= N

    print(metrics)
