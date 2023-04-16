import argparse
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_pkl', type=str, required=True)
    parser.add_argument('--new_pkl', type=str, required=True)
    args = parser.parse_args()

    with open(args.old_pkl, 'rb') as f:
        old = dict(pickle.load(f))

    with open(args.new_pkl, 'rb') as f:
        new = dict(pickle.load(f))

    old_keys = set(old.keys())
    new_keys = set(new.keys())

    for s in old_keys:
        if s not in new_keys:
            print(f'{s} is missing')

    for s in new_keys:
        if s not in old_keys:
            print(f'{s} is extra')

    for s in old_keys:
        if old[s] != new[s]:
            print(f'Mismatch found for {s}')

    print('Finished')


if __name__ == '__main__':
    main()
