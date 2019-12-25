# import from https://gist.github.com/kblackcn/a3f3418dc0e14c633156a4e3843ba566

# example: 
# python3 submit.py -u <id> -p <password> -m <method: OPENMP MPI CUDA> -n <node> -c <ppn> <your code file> <any extra arguments>

import requests, os, sys, time, shutil, getopt

login_url = 'http://49.52.10.141:9002/login'
presubmit_url = 'http://49.52.10.141:9002/preSubmit'
submit_url = 'http://49.52.10.141:9002/submitCode'

R = requests.Session() 

# config

try:
    arguments, values = getopt.getopt(sys.argv[1:], "u:p:m:n:c:", [])
except getopt.error as err:
    print (str(err))
    sys.exit(2)

config = dict()

for currentArgument, currentValue in arguments:
    if currentArgument in ("-u"):
        config['username'] = currentValue
    elif currentArgument in ("-p"):
        config['password'] = currentValue
    elif currentArgument in ("-m"):
        config['method'] = currentValue
    elif currentArgument in ("-n"):
        config['node'] = currentValue
    elif currentArgument in ("-c"):
        config['core'] = currentValue

assert(len(values)>0)

config['file'] = values[0]

config['adv'] = ' '+' '.join(values[1:]) if len(values)>1 else ''

print(config)

# config loaded

try:
    os.mkdir('backup')
except:
    pass

time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

# login

account = {'username' : config['username'],
          'password' : config['password']}

r = R.post(url = login_url, params = account)

if r.text.find('欢迎') == -1:
    print('Login failed.')
    exit()

# pre submit

(_, ext) = os.path.splitext(config['file'])

code_file = [('code', (time_str + ext, open(config['file'], 'rb'), 'text/plain'))]

presubmit_values = {
    'id': 1,
    'compileMethod': config['method']
}

r = R.post(url = presubmit_url, files = code_file, params = presubmit_values)

rj = r.json()

shutil.copy(config['file'], 'backup/%s-%s%s'%(time_str, rj['time_start'], ext))

# submit

submit_values = {
    'compileMethod': config['method'],
    'node': config['node'],
    'ppn': config['core'],
    'id': 1,
    'time_start': rj['time_start'],
    'command_line': rj['command_line'] + config['adv'],
    'path': rj['path']
}

r = R.post(url = submit_url, files = code_file, params = submit_values)

print(r.status_code)

if r.status_code != 200:
    print(r.text)