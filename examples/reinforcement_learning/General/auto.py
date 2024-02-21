
import subprocess

def main():
    different_args = [
        # Run 2: Same setup for acrobot
        {
            '--name': 'fut_5t_6s_balanced',
            '--mode': 'train',
            '--env_type': 'acrobot',
            '--param': 'default'
        },
        # Run 3: Custom setup for pendubot with random parameters
        {
            '--name': 'fut_5t_6s_random',
            '--mode': 'train',
            '--env_type': 'pendubot',
            '--param': 'random'
        },
        # Run 4: Custom setup for acrobot with random parameters
        {
            '--name': 'fut_5t_6s_random',
            '--mode': 'train',
            '--env_type': 'acrobot',
            '--param': 'random'
        },
    ]

    for arg_set in different_args:
        arg_list = ['python', 'main.py']
        for arg, val in arg_set.items():
            arg_list.append(arg)
            arg_list.append(val)
        print(f"Running: {' '.join(arg_list)}")
        result = subprocess.run(arg_list)
        if result.returncode != 0:
            print("Error in the process with args: ", arg_set)
        else:
            print("Finished run with args: ", arg_set)

if __name__ == "__main__":
    main()