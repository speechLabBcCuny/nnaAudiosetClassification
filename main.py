from subprocess import Popen, PIPE

# sp = subprocess.run(["conda","run","-n","speechEnv","python", "test_env.py"],stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
# print(sp.stdout)

#async
process = subprocess.Popen(["conda","run","-n","speechEnv","python", "test_env.py"], stdout=PIPE, stderr=PIPE)
stdout, stderr = process.communicate()
