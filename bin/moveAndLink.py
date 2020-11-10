#!/usr/bin/python3
"""
Move file(s) to new location while leaving a symbolic link at the old location to the new location.

Reads in a file with two or three tab-delimited columns.

In two-column mode:
* First column is current location of file and future location of link to new file
* Second column is new location of file

In three-column mode:
* First column is current location of link to current file, which will be updated to point to new location
* Second column is current location of file
* Third column is new location of file
"""

import argparse
import sys

from pathlib import Path


def main():
    args = parseOpts()

    if args.test:
        test_process_triple()
        return

    num_cols = 2 if args.two_column else 3

    for line in args.infile:
        fields = line.rstrip().split('\t')
        assert len(
            fields
        ) == num_cols, f"Wrong number of columns. Found '{len(fields)}', expected '{num_cols}'"
        if num_cols == 2:
            src = Path(fields[0]).resolve()
            dst = Path(fields[1]).resolve()
            link = src
            if src == dst:
                continue
        else:
            link = Path(fields[0])
            src = Path(fields[1]).resolve()
            dst = Path(fields[2]).resolve()
            assert link.is_link(), f"'{link}' must be symbolic link"
            assert link.samefile(
                src), f"Link '{link}' does not point to '{src}'"

        process_triple(link, src, dst, args.verbose, args.dry_run)


def process_triple(link, src, dst, verbose=False, dry_run=False):
    assert src.exists(), f"File not found: '{src}'"
    assert not dst.exists(), f"Destination exists: '{dst}'"
    #assert not dst.exists() or not src.samefile(dst), f"Source and destination files are the same: '{src}' == '{dst}'"

    # Make directories for dst
    if verbose or dry_run:
        print(f"'{dst.parent}'.mkdir(parents=True, exist_ok=True)")
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)

    # Move src to dst
    if verbose or dry_run:
        print(f"'{src}'.rename('{dst}')")
    if not dry_run:
        src.rename(dst)

    if link.exists() and link.samefile(dst):
        # Moving back to original location, do not delete or create link
        pass
    else:
        # Delete link
        assert not src.exists() or not link.samefile(src)

        if verbose or dry_run:
            print(f"'{link}'.unlink()")
        if not dry_run:
            try:
                link.unlink()
            except FileNotFoundError:
                # Need to do this because unlink doesn't have the
                # missing_ok argument in 3.6 (does in 3.8).  And
                # link.exists() is false because it tests the
                # existence of the target of the link, not the link
                # itself
                pass

        # Make new link
        if verbose or dry_run:
            print(f"'{link}'.symlink_to('{dst}')")
        if not dry_run:
            link.symlink_to(dst)


def make_test_files():
    files = [
        ("test/a/b/c.txt", "I am c"),
        ("test/a/d.txt", "I am d"),
        ("test/e.txt", "I am e"),
    ]

    for f, contents in files:
        p = Path(f)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as fl:
            fl.write(contents)
            print(f"Made '{p}'")


def test_process_triple():
    make_test_files()

    # Basic operation
    triples = [
        ["test/a/b/c.txt", "test/a/b/c.txt", "test/out/c.txt"],
        ["test/a/d.txt", "test/a/d.txt", "test/out/b/c/d.txt"],
        ["test/e.txt", "test/e.txt", "test/out/b/e.txt"],
    ]
    for triple in triples:
        link, src, dst = [Path(p) for p in triple]
        print(link, src, dst)
        process_triple(link, src, dst, verbose=True)

    # Now move files and update links
    triples = [
        ["test/a/b/c.txt", "test/out/c.txt", "test/out2/c.txt"],
        ["test/a/d.txt", "test/out/b/c/d.txt", "test/out2/b/c/d.txt"],
        ["test/e.txt", "test/out/b/e.txt", "test/out2/b/e.txt"],
    ]
    for triple in triples:
        link, src, dst = [Path(p) for p in triple]
        print(link, src, dst)
        process_triple(link, src, dst, verbose=True)

    # Now move files back to where they started
    triples = [
        ["test/a/b/c.txt", "test/out2/c.txt", "test/a/b/c.txt"],
        ["test/a/d.txt", "test/out2/b/c/d.txt", "test/a/d.txt"],
        ["test/e.txt", "test/out2/b/e.txt", "test/e.txt"],
    ]
    for triple in triples:
        link, src, dst = [Path(p) for p in triple]
        print(link, src, dst)
        process_triple(link, src, dst, verbose=True)


def parseOpts():
    parser = argparse.ArgumentParser(
        description='Move files but leave a symbolic link at the old location')
    parser.add_argument('-t',
                        '--test',
                        action="store_true",
                        help='Make test files and exit')
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        help='Print actions taken')
    parser.add_argument('-2',
                        '--two-column',
                        action="store_true",
                        help='Use 2-column mode')
    parser.add_argument('-3',
                        '--three-column',
                        action="store_true",
                        help='Use 3-column mode')
    parser.add_argument('-n',
                        '--dry-run',
                        action="store_true",
                        help="Print what would be done, but don't do it")
    parser.add_argument('infile',
                        nargs='?',
                        type=argparse.FileType('r'),
                        default=sys.stdin,
                        help="Read from files or stdin")
    return parser.parse_args()


if __name__ == '__main__':
    main()
