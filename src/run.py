import subprocess
import platform
import time
import shutil

def run_in_windows(script):
    # start a new cmd window and run the script, keeping it open (/k)
    subprocess.Popen(['start', 'cmd', '/c', f'python {script}'], shell=True)

def run_in_mac(script):
    # tell Terminal.app to run the script in a new window
    subprocess.Popen([
        'osascript', '-e',
        f'tell application "Terminal" to do script "cd {os.getcwd()}; python3 {script}"'
    ])

def run_in_linux(script):
    # try gnome-terminal, then xterm
    if shutil.which('gnome-terminal'):
        subprocess.Popen([
            'gnome-terminal', '--', 'bash', '-c',
            f'python3 {script}; exec bash'
        ])
    elif shutil.which('xterm'):
        subprocess.Popen([
            'xterm', '-hold', '-e', f'python3 {script}'
        ])
    else:
        print("No supported terminal emulator found (gnome-terminal or xterm).")

def main():
    os_name = platform.system()
    runners = {
        'Windows': run_in_windows,
        'Darwin':  run_in_mac,    # macOS
    }
    runner = runners.get(os_name, run_in_linux)

    # launch server.py
    runner('server.py')
    # small pause to let the server start before the client tries to connect
    time.sleep(10)
    # launch client.py
    runner('client.py')

if __name__ == '__main__':
    import os
    main()
