import pandas as pd
import random


def lengthen_project_groups(project_groups):
    num_rows = 0
    for project in project_groups:
        if project.shape[0] > num_rows:
            num_rows = project.shape[0]

    result = []
    for project in project_groups:
        while project.shape[0] < num_rows:
            sampled_row = project.sample(n=1)
            project = project.append(sampled_row)
        result.append(project)

    return result


NUM_PROJECTS = 4
NUM_ARTICLES = 31
NUM_LABELS = 25


def main(directory, project_groups):
    lengthened_groups = lengthen_project_groups(project_groups)
    colnames = []
    for i in (range(NUM_PROJECTS)):
       colnames.append('group_id_' + str(i))
       colnames.append('country_' + str(i))
       colnames.append('project_' + str(i))
       colnames.extend(['article_%d_%d' % (i, j) for j in range(NUM_ARTICLES)])
       colnames.extend(['label_%d_%d' % (i, j) for j in range(NUM_LABELS)])

    rows = []
    for i in range(lengthened_groups[0].shape[0]):
        projects = list(range(NUM_PROJECTS))
        print([lengthened_groups[p].shape for p in range(4)])
        random.shuffle(projects)
        row = []
        for p in projects:
            row.extend(lengthened_groups[p].iloc[i,:].values.tolist())
        rows.append(row)

    merged = pd.DataFrame(rows, columns=colnames)
    merged.to_csv(directory + '/mturk.csv', index=False)


if __name__ == '__main__':
    import sys

    directory = sys.argv[1]
    projects = sys.argv[2:]

    project_groups = []
    for project in projects:
        project_groups.append(pd.read_csv(directory + '/' + project + '/groups.csv'))

    main(directory, project_groups)

