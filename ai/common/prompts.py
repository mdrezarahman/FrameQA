'''
Controls all the prompts - read from disk files, cached
and reloaded on demand
'''
from aparavi import debug
from langchain.prompts import Prompt
from typing import Dict, Any
import glob
import os


class Prompts():
    '''
    This prompts class is a static class which controls all
    the prompts
    '''

    '''
    Privates
    '''
    _prompts: Dict[str, str] = {}
    _init: bool = False

    @staticmethod
    def loadPrompts():
        '''
        This function will read all the prompts and put them into
        the prompts dictionary. The '__prompts' Dict will contain 
        values as string like the following:

        Prompts._prompts['conversation'] = "<text from 'conversion.txt' file>"
        '''
        # Clear the prompts in case we are reloading
        Prompts._prompts = {}

        # Set the root path on where to look
        script_directory = os.path.abspath(os.path.dirname(__file__)) # absolute path of the current file
        debug("The script is located in:" + script_directory)

        # Set the root path to the 'prompts' directory, going one level up and then into 'prompts'
        rootPath = os.path.join(os.path.dirname(script_directory), 'prompts')
        debug("Prompt directory:" + rootPath)

        # Use glob to find all .txt files in the 'prompts' directory and its subdirectories
        promptFiles = glob.glob(os.path.join(rootPath, '**', '*.txt'), recursive=True)

        # Set the root key
        for promptFile in promptFiles:
            # If this is one of the cache files, skip it
            if '__pycache__' in promptFile.lower():
                continue

            # Get the full path
            promptPath = os.path.join(rootPath, promptFile)

            # If this is a directory, skip it
            if not os.path.isfile(promptPath):
                continue

            # Get the basename that returns the filename with extension
            promptName = os.path.basename(promptPath)
            
            try:
                # Remove the .txt
                promptName = os.path.splitext(promptName)[0] if os.path.splitext(promptName)[1] == '.txt' else promptName

            except Exception as e:
                debug("Exception happens during prompt files' extension removal")
                raise e

            # Read the prompt file
            with open(promptPath) as f:
                promptText = f.read()
                f.close()

            # Set it
            Prompts._prompts[promptName] = promptText

    @staticmethod
    def getPrompt(path: str, params: Dict[str, Any] = {}):
        # If we have not loaded the prompts yet, do so now
        if not Prompts._init:
            Prompts.loadPrompts()
            Prompts._init = True

        # Get the components
        components = path.split('.')

        # Walk down the components
        obj: Any = Prompts._prompts
        for component in components:
            if component not in obj:
                raise BaseException(f'Prompt {component} not found in prompts')
            obj = obj[component]

        # Get the prompt builder
        promptBuilder = Prompt.from_template(obj)

        # Setup our prompt
        promptValue = promptBuilder.invoke(params)

        # Convert it to a string
        promptString = promptValue.to_string()

        # Return it
        return promptString
