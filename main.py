from subprocess import Popen, PIPE

# sp = subprocess.run(["conda","run","-n","speechEnv","python", "test_env.py"],stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
# print(sp.stdout)

#async
process = subprocess.Popen(["conda","run","-n","speechEnv","python", "test_env.py"], stdout=PIPE, stderr=PIPE)
stdout, stderr = process.communicate()


# pipe1  outputs a folder
if not os.path.exists(output_folder):
    print("output folder does not exits:\n",output_folder,"\n mp3 file: ",mp3_file_path)
    return False
	os.mkdir(output_folder)
