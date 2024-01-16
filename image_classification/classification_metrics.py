fp = 0
fn = 0
vp = 0
vn = 0
th = 70.00
x = 0
max_val = max(label_pred.values())
            max_lab = [clave for clave, valor in label_pred.items() if valor == max_val][0]
            if max_val >= th:
                if max_lab == label:
                    vp = vp + 1
                    vpl = vpl + 1
                else:
                    fp = fp + 1
                    fpl = fpl + 1
            else:
                if label in label_pred:
                    fn = fn + 1
                    fnl = fnl + 1
                else:
                    vn = vn + 1
                    vnl = vnl + 1
        print(label + "; " + " FN: " + str(fnl) + " FP: " + str(fpl) + " VN: " + str(vnl) + " VP: " + str(vpl))


print("Falsos negativos: " + str(fn))
    print("Verdaderos positivos: " + str(vp))
    print("Falsos positivos: " + str(fp))
    print("Verdaderos negativos: " + str(vn))

    print("Accuracy: " + str(calcular_efectividad(vp, fp, fn)))
    print("Precision: " + str(calcular_precision(vp, fp)))
    print("F1: " + str(calcular_f1_score(vp, fp, fn)))
    FRR = calcular_frr(vp, fn)
    FAR = calcular_far(fp, vn)
    print("FRR: " + str(FRR))
    print("FAR: " + str(FAR))
    print("EER: " + str(calcular_eer(FRR, FAR)))