import os
import re
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log', type=str, nargs='+')
    args = parser.parse_args()

    if not isinstance(args.log, list):
        args.log = [args.log]

    for log in args.log:
        with open(log, 'r') as logfile:
            text = logfile.read()
            ret = re.search("==> model.final_name: (.*)\n", text)
            if ret is None:
                print("No model.final_name in log file: {}. Skipping...".format(log))
                continue
            name = ret.group(1)

        dirname = os.path.dirname(log)
        new_path = os.path.join(dirname, "{}.log".format(name))
        os.rename(log, new_path)

if __name__ == '__main__':
    main()
