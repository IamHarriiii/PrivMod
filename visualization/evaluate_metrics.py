from sklearn.metrics import classification_report

y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 0]

print(classification_report(y_true, y_pred))