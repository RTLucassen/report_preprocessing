"""
This module contains the LabelStudioClient class which is used to interact with the Label Studio instance.
"""

from label_studio_sdk.client import Client
from getpass import getpass

class LabelStudioClient:

    def __init__(self, api_key, url) -> None:
        if url is None:
            print("Label Studio URL not set as an environment variable")
            url = input("Enter the URL of the Label Studio instance: ")
        
        # secret input
        if api_key is None:
            print("Label Studio API key not set as an environment variable")
            api_key = getpass("Enter the API key of the Label Studio instance: ")

        # Create a client
        self.client = Client(url=url, api_key=api_key)
    
    def check_connection(self):
        try:
            self.client.check_connection()
            print("Connected to the Label Studio instance.")
        except Exception as e:
            print(f"Could not connect to the Label Studio instance. Error: {e}")
            return None
        
    def get_project(self, project_id):
        try:
            project = self.client.get_project(project_id)
            print(f"Project with id {project_id} found.")
            return project
        except Exception as e:
            print(f"Could not find the project with id {project_id}. Error: {e}")
            return None