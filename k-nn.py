import numpy as np
from scipy.spatial import distance

def KNN(X_train, y_train, X_test, k):
    predictions = []
    
    for x_test in X_test:
        distances = []
        
        for x_train in X_train:
            # محاسبه فاصله اقلیدسی بین داده آزمایشی و داده‌های آموزشی
            dist = distance.euclidean(x_test, x_train)
            distances.append(dist)
        
        # پیدا کردن k نزدیک‌ترین همسایه
        nearest_neighbors = np.argsort(distances)[:k]
        
        # برای محاسبه تعداد هر کلاس در همسایه‌ها استفاده می‌شود
        class_votes = np.zeros(max(y_train) + 1)
        for neighbor in nearest_neighbors:
            class_votes[y_train[neighbor]] += 1
        
        # کلاس با بیشترین تعداد رای برای پیش‌بینی انتخاب می‌شود
        predicted_class = np.argmax(class_votes)
        predictions.append(predicted_class)
    
    return predictions
