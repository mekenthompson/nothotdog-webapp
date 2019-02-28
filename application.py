import os, sys
import uuid
from flask import Flask, render_template, request

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

try:  
   os.environ["PREDICTION_KEY"]
except KeyError: 
   print ("*** ENVVAR Missing: Please set the environment variable PREDICTION_KEY ***")
   sys.exit(1)

try:  
   os.environ["TRAINING_KEY"]
except KeyError: 
   print ("*** ENVVAR Missing: Please set the environment variable TRAINING_KEY ***")
   sys.exit(1)

TRAINING_KEY = os.environ["TRAINING_KEY"]
PREDICTION_KEY= os.environ["PREDICTION_KEY"]
ENDPOINT = "https://australiaeast.api.cognitive.microsoft.com"
SAMPLE_PROJECT_NAME = "nothotdog-classifier"

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = str(uuid.uuid1())
        destination = "./" + filename
        print(destination)
        file.save(destination)
        results = predict_project(filename)
        # delete here
        os.remove(filename)

        #Create dictionary of result tags and probabilities
        result_dict = {}
        for prediction in results.predictions:
            result_dict[prediction.tag_name] = prediction.probability
        
    return render_template("complete.html", result_dict=result_dict)

def find_project():
    # Use the training API to find the SDK sample project created from the training example.
    trainer = CustomVisionTrainingClient(TRAINING_KEY, endpoint=ENDPOINT)

    for proj in trainer.get_projects():
        if (proj.name == SAMPLE_PROJECT_NAME):
            return proj
    
    return False
    
def predict_project(filename):
    predictor = CustomVisionPredictionClient(PREDICTION_KEY, endpoint=ENDPOINT)

    # Find or train a new project to use for prediction.
    project = find_project()
    if (project == False):
        print("Run training first, project not found!")
        return []

    with open(filename, mode="rb") as test_data:
        results = predictor.predict_image(project.id, test_data.read())

    # Display the results.
    for prediction in results.predictions:
        print ("\t" + prediction.tag_name + ": {0:.2f}%".format(prediction.probability * 100))
    
    return results

if __name__=="__main__":
    app.run()
