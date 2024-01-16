def calcular_efectividad(TP, FP, FN):
    total = TP + FP + FN
    efectividad = TP / total
    return efectividad

def calcular_precision(TP, FP):
    precision = TP / (TP + FP)
    return precision

def calcular_f1_score(TP, FP, FN):
    precision = calcular_precision(TP, FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def calcular_eer(FRR, FAR):
    eer = (FRR + FAR) / 2
    return eer

def calcular_frr(TP, FN):
    frr = FN / (TP + FN)
    return frr

def calcular_far(FP, TN):
    far = FP / (FP + TN)
    return far