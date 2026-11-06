import tkinter as tk
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto', return_X_y=True)
X = X / 255.0

X_train, y_train = X[:5000], y[:5000]
X_test, y_test = X[5000:6000], y[5000:6000]

models = {
    'Logistic Regression': LogisticRegression(max_iter=100),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=30),
    'SVM': SVC()
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results.append({'Model': name, 'Accuracy': f"{acc * 100:.1f}%"})

print(pd.DataFrame(results))

root = tk.Tk()
root.title("MNIST Canvas")

canvas = tk.Canvas(root, width=280, height=280, bg='black')
canvas.pack(pady=10)

img = Image.new('L', (280, 280), 0)
draw = ImageDraw.Draw(img)

def paint(event):
    r = 10
    canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill='white', outline='white')
    draw.ellipse([event.x - r, event.y - r, event.x + r, event.y + r], fill=255)

canvas.bind('<B1-Motion>', paint)

result_label = tk.Label(root, text="Draw a digit and click Predict", font=('Arial', 12))
result_label.pack(pady=10)

def predict():
    data = np.array(img.resize((28, 28))) / 255.0
    data = data.reshape(1, -1)
    
    text = ""
    for name, model in models.items():
        text += f"{name}: {model.predict(data)[0]}\n"
    result_label.config(text=text)

def clear():
    canvas.delete('all')
    draw.rectangle([0, 0, 280, 280], fill=0)
    result_label.config(text="Draw a digit and click Predict")

btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="Predict", command=predict, width=10).pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="Clear", command=clear, width=10).pack(side=tk.LEFT, padx=5)

root.mainloop()

