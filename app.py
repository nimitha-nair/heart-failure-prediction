from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]

        prediction = model.predict(final_features)

        if prediction[0] == 1:
            result = "🔴 Risk of heart failure"
        else:
            result = "🟢 No risk of heart failure"

        return render_template('index.html', prediction_text=result)

    return render_template('index.html', prediction_text=None)

if __name__ == "__main__":
    app.run(debug=True)
