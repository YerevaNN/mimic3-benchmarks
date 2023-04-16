import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_listfile', type=str, required=True)
    parser.add_argument('--new_listfile', type=str, required=True)
    parser.add_argument('--sort', dest='sort', action='store_true')
    parser.set_defaults(sort=False)
    args = parser.parse_args()

    with open(args.old_listfile, 'r') as f:
        old_listfile = f.readlines()
    if args.sort:
        old_listfile = sorted(old_listfile)

    with open(args.new_listfile, 'r') as f:
        new_listfile = f.readlines()
    if args.sort:
        new_listfile = sorted(new_listfile)

    assert len(old_listfile) == len(new_listfile)

    for (old, new) in zip(old_listfile, new_listfile):
        if old.strip() != new.strip():
            print('Mismatch found:')
            print('\told:', old)
            print('\tnew:', new)


if __name__ == '__main__':
    main()
