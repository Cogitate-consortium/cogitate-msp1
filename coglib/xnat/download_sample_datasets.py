#!/usr/bin/env python
# author: Praveen Sripad <praveen.sripad@ae.mpg.de>

"""Simple script uses xnatpy to download all or part of the Cogitate Sample Dataset.

The cogitate sample dataset consists of data shared from their original projects.

Two subjects from every site are included:
FMRI - CC102, CC202, CD101, CD199
MEEG - CA124, CA140, CB013, CB072
ECOG - CE107, CE110, CF102, CF104

"""

import os
import os.path as op
import xnat


def _download_resources_recursively(xnat_object, download_dir):
    """Download xnat resources recursively for projects, subjects and experiments.

    Scan level data/resources are not downloaded.

    """
    print('Downloaded data for %s into directory %s'
          % (xnat_object, download_dir))

    if not op.isdir(download_dir):
        os.mkdir(download_dir)

    resources_list = xnat_object.resources.listing

    if len(resources_list) != 0:
        for res in resources_list:
            print(res)
            res.download_dir(download_dir)

    # if xnat object is project, recurse
    if type(xnat_object).__name__ == 'ProjectData':
        for subj in xnat_object.subjects.listing:
            _download_resources_recursively(subj,
                                            op.join(download_dir, subj.label))

    # if xnat object is subject, recurse
    if type(xnat_object).__name__ == 'SubjectData':
        for exp in xnat_object.experiments.listing:
            _download_resources_recursively(exp, op.join(download_dir,
                                            xnat_object.label, exp.label))

    return


# CONFIG
xnat_host = 'https://xnat-curate.ae.mpg.de'
netrc_file = op.join(op.expanduser('~'), '.xnat_curate_netrc')

myproject = 'cogitate_sample_dataset'
download_dir = op.join(op.expanduser('~'), 'Downloads')
# CONFIG

# start connection to XNAT
connection = xnat.connect(xnat_host, netrc_file=netrc_file)

# choose the project
project = connection.projects[myproject]

# download everything under the project
project.download_dir(download_dir)

# download all the data for a subject
# mysubject = 'CA124'
# project.subjects[mysubject].download_dir(download_dir)

# download all the resources only (excluding the scans)
# _download_resources_recursively(project, download_dir)
