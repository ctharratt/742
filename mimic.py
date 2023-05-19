import openai
import os
import time

openai.api_key =  "" #Set your OpenAI api key here 

path_to_users = "/Users/samanthatang/Documents/UMD/Grad_School/Spring_2023/742/Reddit_Cross-Topic-AV-Corpus_(1000_users)" # Set this to your path to the Reddit Dataset
output_directory = "/Users/samanthatang/Documents/UMD/Grad_School/Spring_2023/742/gpt_texts" # Set this to the path to the folder you would like mimics sent to
# Switch file path to location of Reddit Dataset, get a list of the items in the Dataset directory, then sort the list alphabetically
os.chdir(path_to_users)
dir = os.listdir()
dir_sorted = sorted(dir, key=str.lower)
# Loop through each item in the dataset (At this point, these should be folders for each author)
for i in range(0,len(dir_sorted)):
    # Check you are at a folder as there are other items in the dataset directory
    folder = dir_sorted[i]  
    if os.path.isdir(f"{path_to_users}/{folder}"):
        # Sleep to not overload the model
        time.sleep(10)
        # Move into the specified author's folder, get a list of the items in the folder, then sort the list alphabetically
        os.chdir(f"{path_to_users}/{folder}")
        folder_dir = os.listdir()
        fdir_sorted = sorted(folder_dir, key=str.lower)
        # Pull the 4th item in the folder as it is the 4th known document we will use for evaluation
        file = fdir_sorted[3]
        topic = "[" + file.split("[")[1]
        gpt_file = "gpt_" + folder + topic + ".txt"
        # Form a name for the mimic file
        mimic_check = f"{output_directory}/{gpt_file}"
        if (not os.path.isfile(mimic_check)):
            file_path = f"{path_to_users}/{folder}/{file}"
            # Form the prompt
            with open(file_path, "r") as file:
                prompt = "The following was written by Charlie. \""  
                # Copy in the first 8000 bytes of the text written by the original author
                prompt = prompt + file.read(8000)
                prompt = prompt + "\" Please mimic Charlie's writing style and write on the same topic. You must write at least 800 words."
                # The following are the variations for prompts that could be applied, comment out the above line and uncomment the one you wish to use
                #prompt = prompt + "\" Do not respond to Charlie. However, please mimic Charlie's writing style and write on the same topic. You must write at least 800 words."
                #prompt = prompt + "\" Can you write ramblings in the style of Charlie on the same topics. You must write at least 800 words. "
                #prompt = prompt + "\" DO NOT respond to Charlie. Instead, write in the style of Charlie on the same topics using at least 800 words."
                #prompt = prompt + "\" Please mimic Charlie's writing style and write on the same topic. DO NOT RESPOND TO CHARLIE. JUST ACT AS IF YOU ARE HIM. You must write at least 800 words."
            # Send the formed prompt to GPT
            response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages = [{"role": "user", "content": prompt}],
                    max_tokens = 2000,
                    temperature = 0)
            print("Finished:", folder)
            # Clean up the response by removing new lines and extraneous white space
            text = response.choices[0].message.content
            text = os.linesep.join([s for s in text.splitlines() if s]).replace("\n", " ")
            # Move into the directory to save mimics in and save the generated response there
            os.chdir(output_directory)
            with open(gpt_file, 'a') as f:
                f.write(text)