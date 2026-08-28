# Third-party software and acknowledgments

## oculus-python

The optional raw-data conversion step invokes the external
[`oculus-python`](https://gitlab.gbar.dtu.dk/fletho/oculus-python) package:

- package version used for this release: `0.0.2a1`;
- author and copyright holder: Fletcher Thompson, Copyright 2024;
- license: Apache License 2.0;
- package page: <https://pypi.org/project/oculus-python/0.0.2a1/>.

`oculus-python` reads Blueprint Subsea Oculus log files and provides the
`bps_oculus_io` NetCDF4 exporter called by this repository. Its source code is
not copied into this repository and its installed package files are not
modified by the wrapper.

We thank Fletcher Thompson and the contributors to `oculus-python` for making
the Oculus log reader and export tools publicly available.

The MIT license in this repository applies only to the original code contained
here. Third-party software remains subject to its own license terms.
