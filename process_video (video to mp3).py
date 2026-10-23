# convert the video to mp3
import os
import subprocess

files = os.listdir('Videos')

for file in files:
    # Sirf mp4 files process karein
    if not file.endswith('.mp4'):
        continue
    
    # Extension hatayein (.mp4)
    name_without_ext = file.replace('.mp4', '')
    
    # " - " se split karein (eg: ['Lec 01', 'Goals of the Course...'])
    parts = name_without_ext.split(' - ')
    
    # "Lec 01" me se number '01' nikalne ke liye
    tutorial_number = parts[0].replace('Lec ', '').strip()
    
    # Baaki bacha hua title
    file_name = parts[1].strip() if len(parts) > 1 else name_without_ext
    input_path = os.path.join('Videos', file)
    output_path = f"audios/{tutorial_number}_{file_name}.mp3"
    print(tutorial_number, file_name)
    subprocess.run(["ffmpeg", "-i", input_path, output_path])