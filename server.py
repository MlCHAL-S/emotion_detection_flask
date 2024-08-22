"""
Server running a web app.
"""

from flask import Flask, render_template, request
from EmotionDetection.emotion_detection import emotion_detector

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles the main route for the Emotion Detector."""
    if request.method == 'POST':
        text = request.form.get('textToAnalyze', '')

        result = emotion_detector(text)
        if result['dominant_emotion'] is None:
            result['error'] = 'Invalid text! Please try again!'

        return render_template('index.html', result=result)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
