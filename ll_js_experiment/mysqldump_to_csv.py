"""
Given wikipedia page and categorylinks mysql dump file, convert them to csv file

Author: Yuren 'Rock' Pang
Reference: James Mishra, link: https://github.com/jamesmishra/mysqldump-to-csv

"""

import csv
import sys
import logging

# This prevents prematurely closed pipes from raising
# an exception in Python
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE, SIG_DFL)

# allow large content in the dump
csv.field_size_limit(sys.maxsize)


def is_insert(line):
    """
    Returns true if the line begins a SQL insert statement.
    """
    return line.startswith('INSERT INTO') or False


def get_values(line):
    """
    Returns the portion of an INSERT statement containing values
    """
    return line.partition('` VALUES ')[2]


def values_sanity_check(values):
    """
    Ensures that values from the INSERT statement meet basic checks.
    """
    assert values
    assert values[0] == '('
    # Assertions have not been raised
    return True


def parse_values(values, outfile, column_kept):
    """
    Given a file handle and the raw values from a MySQL INSERT
    statement, write the equivalent CSV to the file
    """
    latest_row = []

    reader = csv.reader([values], delimiter=',',
                        doublequote=False,
                        escapechar='\\',
                        quotechar="'",
                        strict=True
    )

    writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)
    for reader_row in reader:
        for column in reader_row:
            # If our current string is empty...
            if len(column) == 0 or column == 'NULL':
                latest_row.append(chr(0))
                continue
            # If our string starts with an open paren
            if column[0] == "(":
                # Assume that this column does not begin
                # a new row.
                new_row = False
                # If we've been filling out a row
                if len(latest_row) > 0:
                    # Check if the previous entry ended in
                    # a close paren. If so, the row we've
                    # been filling out has been COMPLETED
                    # as:
                    #    1) the previous entry ended in a )
                    #    2) the current entry starts with a (
                    if latest_row[-1][-1] == ")":
                        # Remove the close paren.
                        latest_row[-1] = latest_row[-1][:-1]
                        new_row = True
                # If we've found a new row, write it out
                # and begin our new one
                if new_row:
                    writer.writerow(latest_row)
                    latest_row = []
                # If we're beginning a new row, eliminate the
                # opening parentheses.
                if len(latest_row) == 0:
                    column = column[1:]
            # Add our column to the row we're working on.
            latest_row.append(column)
        # At the end of an INSERT statement, we'll
        # have the semicolon.
        # Make sure to remove the semicolon and
        # the close paren.
        if latest_row[-1][-2:] == ");":
            latest_row[-1] = latest_row[-1][:-2]
            writer.writerow(latest_row)


def is_hidden():
    return False


def create_csv_from_dump(dump_file, dest_csv_file, column_kept):
    """
    Parse arguments and start the program
    """
    # Iterate over all lines in all files
    # listed in sys.argv[1:]
    # or stdin if no args given.

    try:
        with open(dump_file, 'r', encoding="UTF-8", errors="replace") as file, \
                open(dest_csv_file, 'w', encoding="UTF-8") as dest:
            for line in file.readlines():
                if is_insert(line):
                    values = get_values(line)
                    if values_sanity_check(values):
                        parse_values(values, dest, column_kept)
    except KeyboardInterrupt:
        sys.exit(0)


def main(map_directory, pages_sql, categorylinks_sql):
    pages_path = map_directory + "/wiki_pages.csv"
    categorylinks_path = map_directory + "/wiki_categorylinks.csv"

    create_csv_from_dump(pages_sql, pages_path, [0, 2])
    logging.info("%s has been processed. Stored in %s", pages_sql, map_directory)
    create_csv_from_dump(categorylinks_sql, categorylinks_path, [0, 1])


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)

    map_directory = "../" + sys.argv[1]
    pages_sql = sys.argv[2]
    logging.info("%s has been processed. Stored in %s", pages_sql, map_directory)

    categorylinks_sql = sys.argv[3]
    logging.info("%s has been processed. Stored in %s", categorylinks_sql, map_directory)

    main(map_directory, pages_sql, categorylinks_sql)