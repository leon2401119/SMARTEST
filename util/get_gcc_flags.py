import subprocess

INVALID_CHAR = ['=','.']

f = open('../flags','w')

command = ['gcc','--help=optimizers']
output = subprocess.check_output(command).decode('utf-8')
prefix = '-f'
for i in range(len(output)-1):
    if output[i:i+len(prefix)] == prefix:
        opt = ''
        current = i
        while True:
            if output[current] in INVALID_CHAR:
                break

            elif output[current]==' ':
                print(opt,file=f)
                #f.write(opt)
                break

            else:
                opt += output[current]
                current += 1

f.close()
