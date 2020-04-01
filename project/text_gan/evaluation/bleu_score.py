from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def bleu1(refs, hypo):
    return corpus_bleu(
        refs, hypo, weights=[1])


def bleu4(refs, hypo):
    return corpus_bleu(
        refs, hypo, weights=[0, 0, 0, 1],
        smoothing_function=SmoothingFunction().method4)
