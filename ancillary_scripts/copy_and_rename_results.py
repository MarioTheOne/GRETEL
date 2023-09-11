import os
import shutil

def move_and_rename_json_files(root_folder):
    counter = 0
    
    for subfolder, _, files in os.walk(root_folder):
        for filename in files:
            if filename.endswith('.json'):
                json_path = os.path.join(subfolder, filename)
                new_filename = f'run_fold_{counter}.json'
                new_path = os.path.join(root_folder, new_filename)
                
                shutil.move(json_path, new_path)
                
                counter += 1
                shutil.rmtree(subfolder)


folder_path = input("Enter the path of the root folder: ")
# Iterate through all files in the folder
folder_names = os.listdir(folder_path)
for foldername in folder_names:
    move_and_rename_json_files(os.path.join(folder_path, foldername))

print("JSON files moved and renamed successfully.")