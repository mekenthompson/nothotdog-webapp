import os
import time

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient

TRAINING_KEY = "1485d2b010f64c41a44a9eb4b8052697"
SAMPLE_PROJECT_NAME = "blabla"

ENDPOINT = "https://australiaeast.api.cognitive.microsoft.com"

IMAGES_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "images")

def train_project():

    trainer = CustomVisionTrainingClient(TRAINING_KEY, endpoint=ENDPOINT)

    # Create a new project
    print ("Creating project...")
    project = trainer.create_project(SAMPLE_PROJECT_NAME)

    # Make two tags in the new project
    hemlock_tag = trainer.create_tag(project.id, "Hemlock")
    cherry_tag = trainer.create_tag(project.id, "Japanese Cherry")

    print ("Adding images...")
    hemlock_dir = os.path.join(IMAGES_FOLDER, "Hemlock")
    for image in os.listdir(hemlock_dir):
        with open(os.path.join(hemlock_dir, image), mode="rb") as img_data: 
            trainer.create_images_from_data(project.id, img_data.read(), [ hemlock_tag.id ])
    
    cherry_dir = os.path.join(IMAGES_FOLDER, "Japanese Cherry")
    for image in os.listdir(cherry_dir):
        with open(os.path.join(cherry_dir, image), mode="rb") as img_data: 
            trainer.create_images_from_data(project.id, img_data.read(), [ cherry_tag.id ])

    print ("Training...")
    iteration = trainer.train_project(project.id)
    while (iteration.status == "Training"):
        iteration = trainer.get_iteration(project.id, iteration.id)
        print ("Training status: " + iteration.status)
        time.sleep(1)

    # The iteration is now trained. Make it the default project endpoint
    trainer.update_iteration(project.id, iteration.id, is_default=True)
    print ("Done!")
    return project

if __name__ == "__main__":
    train_project()