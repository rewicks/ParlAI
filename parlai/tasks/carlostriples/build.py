#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Download and build the data if it does not exist.

from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
import os

RESOURCES = [
	DownloadableFile(
		'0B06gib_77EnxNE5OdHNfaVR1U0U',
		'MovieTriples.Dataset.tar',
		'e8fbc0027e54e0a916abd9c969eb35f708ed1467d7ef4e3b17a56739d65cb200',
		from_google=True,
	),
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'carlostriples')
    version = 'None'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
