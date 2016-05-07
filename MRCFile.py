"""MRCFile - read and write image data in the MRC file format.

Example:

    Reading

        from MRCFile import MRCFile

        # Setup file object
        f = MRCFile('zika.mrc')

        # Actually load volume slices from disk
        f.load_all_slices()

        # Data is now available in the 3d array f.slices, e.g.,
        display_image(f.slices[:,:,0])

    Writing

        my_volume_processing(f.slices)

        f.write_file('zika_updated.mrc')

Please send bugs and comments to dynerman@berkeley.edu

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import sys
import struct
import timeit
import scipy as sp
import numpy as np
import os.path
from collections import namedtuple

class MRCFile:
    header_format = [( "i", "nx" ),        # 0 volume/image x size
                             ( "i", "ny" ),        # 1 volume/image y size
                             ( "i", "nz" ),        # 2 volume z size/stack # of images
                             ( "i", "mode" ),       # 3 volume/image data type, 0=char, 1=short, 2=float
                             ( "i", "nxStart" ),   # 4 unit cell offset
                             ( "i", "nyStart" ),   # 5
                             ( "i", "nzStart" ),   # 6
                             ( "i", "mx" ),        # 7 unit cell size in voxels
                             ( "i", "my" ),        # 8
                             ( "i", "mz" ),        # 9
                             ( "f", "a" ),         # 10 unit cell dimensions in Angstroms
                             ( "f", "b" ),         # 11
                             ( "f", "c" ),         # 12
                             ( "f", "alpha" ),     # 13 unit cell angle in degrees
                             ( "f", "beta" ),      # 14
                             ( "f", "gamma" ),    # 15
                             ( "i", "mapc" ),      # 16 column axis
                             ( "i", "mapr" ),      # 17 row axis
                             ( "i", "maps" ),      # 18 section axis
                             ( "f", "amin" ),      # 19 minimum density value
                             ( "f", "amax" ),      # 20 maximum density value
                             ( "f", "amean" ),    # 21 average density value
                             ( "i", "ispg" ),       # 22 space group number
                             ( "i", "nsymbt" ),    # 23 bytes used for sym. ops. table
                             ( "25f", "extra" ),   # 24 user-defined info
                             ( "f", "xOrigin" ),    # 49 phase origin in pixels
                             ( "f", "yOrigin" ),    # 50
                             ( "f", "zOrigin" ),    # 51
                             ( "4s", "map" ),      # 52 identifier for map file ("MAP ")
                             ( "4s", "machst" ),   # 53 machine stamp
                             ( "f", "arms" ),       # 54 RMS deviation
                             ( "i", "nlabl" ),       # 55 number of labels usedels
                             ( "800s", "labels" )]  # 56-255 10 80-char labels

    header = None

    # @profile
    def __init__(self, filename):
        self.mrc_fd = open(filename, 'r')

        if MRCFile.header is None:
            MRCFile.header = namedtuple( 'header', [ field[1] for field in MRCFile.header_format ] )

        header_format_string = "".join([ field[0] for field in self.header_format ])
        self.header_size = struct.calcsize(header_format_string)

        assert self.header_size == 1024

        header_data = struct.unpack(header_format_string, self.mrc_fd.read(self.header_size))

        # Annoying hack: field 24 in header data "extra" needs to be
        # passed as a list to the namedtuple
        # constructor. struct.unpack returns the header sequentially,
        # so we have to repack it before sending it to namedtuple
        header_data = header_data[:24] + (header_data[24:49],) + header_data[49:]

        self.header = MRCFile.header(*header_data)

        self.num_slice_elements = self.header.nx*self.header.ny

        assert self.header.mode >= 0 and self.header.mode <= 2

        if self.header.mode == 0:
            # slice data is unsigned char
            self.slice_element_type = np.uint8
            self.slice_element_format_string = 'B'
        elif self.header.mode == 1:
            # slice data is short
            self.slice_type = np.int16
            self.slice_element_format_string = 'h'
        elif self.header.mode == 2:
            # slice data is float
            self.slice_type = np.float32
            self.slice_element_format_string = 'f'

        self.slice_element_byte_size = np.dtype(self.slice_type).itemsize

        self.vol = self.header.nz*self.header.nx*self.header.ny

        assert self.vol > 0

    # @profile
    def write_file(self,filename,overwrite=False):
        # Write MRC data to filename

        if os.path.exists(filename) and not overwrite:
            print("%s exists, call write_file(%s,overwrite=True) to overwrite"%(filename,filename), file=sys.stderr)
            return

        assert self.header.nsymbt == 0, "Writing MRC files with symmetry operations not supported"

        out_fd = open(filename, 'wb')

        # Write header
        raw_header_data = self.header._asdict().values()

        # Annoying hack: field 24 in header data "extra" needs to be
        # passed unpacked to struct.pack
        header_data = raw_header_data[:24] + [e for e in raw_header_data[24]] + raw_header_data[25:]

        header_format_string = "".join([ field[0] for field in self.header_format ])

        out_fd.write(struct.pack(header_format_string,*header_data))

        self.slices.flatten().astype(self.slice_type).tofile(out_fd)

        out_fd.close()

    # @profile
    def load_all_slices(self):
        self.mrc_fd.seek(self.header_size + self.header.nsymbt)

        self.slices = np.ndarray((self.header.nz, self.header.nx, self.header.ny, self.header.nz))

        all_format_string = '%d%s'%(self.vol, self.slice_element_format_string)

        all_byte_size = self.slice_element_byte_size*self.vol
        self.slices = np.array(struct.unpack(all_format_string, self.mrc_fd.read(all_byte_size)))
        self.slices = self.slices.reshape((self.header.nz, self.header.nx, self.header.ny))

    # @profile
    def get_images(self, image_indices=None):
        if image_indices is None:
            image_indices = range(self.header.nz)

        self.images = np.ndarray((len(image_indices), self.header.nx, self.header.ny))

        for image_index, image in enumerate(image_indices):
            self.mrc_fd.seek(self.header_size + self.header.nsymbt + image_index*self.num_cols*self.num_rows, 0)

            slice_format_string = '%d%d'%(self.num_slice_elements,self.slice_element_format_string)
            slice_byte_size = self.slice_element_byte_size*self.num_cols*self.num_rows
            images[image_index,:,:] = np.array(struct.unpack('<%df'%(self.num_cols*self.num_rows), self.mrc_fd.read(slice_byte_size))).reshape((self.num_rows,self.num_cols))

        return images

    # @profile
    def close(self):
        self.mrc_fd.close()
        self.mrc_fd = None

