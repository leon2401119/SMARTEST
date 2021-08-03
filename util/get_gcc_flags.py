import subprocess

INVALID_CHAR = ['=','.',':']

f = open('../flags','w')

command = ['gcc','--help=optimizers']
output = subprocess.check_output(command).decode('utf-8')
prefix = '-f'
cursor = 0
while True:
    if cursor >= len(output)-2:
        break
    if output[cursor:cursor+2] == prefix:
        opt = ''
        while True:
            if output[cursor] in INVALID_CHAR:
                break

            elif output[cursor]==' ':
                # this flag disables /causes trouble for other flags
                if opt != '-flive-patching':
                    print(opt,file=f)
                #f.write(opt)
                break

            else:
                opt += output[cursor]
                cursor += 1
    cursor += 1

f.close()
