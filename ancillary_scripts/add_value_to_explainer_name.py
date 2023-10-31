import os
import json
import jsonpickle

# Folder containing the JSON files
root_folder = input("Enter the path of the root folder: ")
field_to_add = 'number_of_cycles'

# Iterate through all files in the folder
# file_names = os.listdir(folder_path)
# for filename in file_names:

# Iterate through all files in the subfolders of the root folder
for subfolder, _, files in os.walk(root_folder):
    for filename in files:  
        if filename.endswith('.json'):
            json_file_path = os.path.join(subfolder, filename)
            
            data_to_write = None
            # Read the JSON file
            with open(json_file_path, 'r') as json_file:
                r_data = json_file.read()
                data = jsonpickle.decode(r_data)
                # data = json.load(json_file)

                old_name = data['config']['explainer']['name']
                posfix = data['config']['dataset']['parameters'][field_to_add]

            
                # Change the value of the "name" field
                data['config']['explainer']['name'] = f'{old_name}-{posfix}'
                data_to_write = jsonpickle.encode(data)
            
            # Save the changes back to the JSON file
            with open(json_file_path, 'w') as json_file:
                json_file.write(data_to_write)
                # json.dump(data, json_file, indent=4)

            print(f'Changed the "name" field in {filename} to: {old_name}-{posfix}')
        
        






