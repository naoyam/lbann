////////////////////////////////////////////////////////////////////////////////
//// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
//// Produced at the Lawrence Livermore National Laboratory.
//// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
//// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
////
//// LLNL-CODE-697807.
//// All rights reserved.
////
//// This file is part of LBANN: Livermore Big Artificial Neural Network
//// Toolkit. For details, see http://software.llnl.gov/LBANN or
//// https://github.com/LLNL/LBANN.
////
//// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
//// may not use this file except in compliance with the License.  You may
//// obtain a copy of the License at:
////
//// http://www.apache.org/licenses/LICENSE-2.0
////
//// Unless required by applicable law or agreed to in writing, software
//// distributed under the License is distributed on an "AS IS" BASIS,
//// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
//// implied. See the License for the specific language governing
//// permissions and limitations under the license.
////
//////////////////////////////////////////////////////////////////////////////////
//TODO:
//write it so it takes in the appropriate number of directories
//adjust load so that it loads in a label
//use the right tag for that label
//
#include "lbann/data_readers/data_reader_hdf5.hpp"
#include "lbann/utils/profiling.hpp"
#include <cstdio>
#include <string>
#include <fstream>
#include <unordered_set>
#include <iostream>
#include "dirent.h"
#include <cstring>
#include "lbann/utils/distconv.hpp"
namespace lbann {
    const std::string hdf5_reader::HDF5_KEY_DATA = "full";
    const std::string hdf5_reader::HDF5_KEY_RESPONSES = "redshifts";

    hdf5_reader::hdf5_reader(const bool shuffle)
        : generic_data_reader(shuffle) {
          }
    // helper function, couldnt find this in the std lib
   // bool file_ends_with(const std::string &mainStr, const std::string &toMatch)
   // {
   //     return (mainStr.size() >= toMatch.size() &&
   //             mainStr.compare(mainStr.size() - toMatch.size(), toMatch.size(), toMatch) == 0);
   // }
    // collect all files names in a directory, ignore all files that don't end in .hdf5
    // this should possibly be changes to "or .h5" as I think that is a valid ending to hdf5 files
    //std::vector<std::string> get_filenames(std::string dir_path) {
    //    std::vector<std::string> file_names;
    //    DIR *dir = opendir(dir_path.c_str());
    //   struct dirent *entry;
    //    std::string file_ending = ".hdf5";
    //    while ((entry = readdir(dir)) != NULL) {
    //        std::string temp_path = dir_path;
    //        std::string entry_name = entry->d_name;

    //        if(file_ends_with(entry_name, file_ending)) {
    //            file_names.push_back(temp_path.append(entry_name));                                                 
    //        }
    //    }
    //    closedir(dir);
    //    return file_names;
    //}

    void hdf5_reader::read_hdf5(hsize_t h_data, hsize_t filespace, int rank, std::string key, hsize_t* dims, DataType * data_out) {
        // this is the splits, right now it is hard coded to split along the z axis
        int num_io_parts = dc::get_number_of_io_partitions();
        int ylines = 1;
        int xlines = 1;
        int zlines = num_io_parts;
        int channellines = 1;

        // todo: when taking care of the odd case this cant be an int
        int xPerNode = dims[0]/xlines;
        int yPerNode = dims[1]/ylines;
        int zPerNode = dims[2]/zlines;
        int cPerNode = dims[3]/channellines;
       	//TODO: change this to be allocated elsewhere
        short int data_out[xPerNode*yPerNode*zPerNode*cPerNode]; 
        hsize_t dims_local[4];
        hsize_t offset[4];
        hsize_t count[4];
        // local dimensions aka the dimensions of the slab we will read in
        dims_local[0] = xPerNode;
        dims_local[1] = yPerNode;
        dims_local[2] = zPerNode;
        dims_local[3] = cPerNode;
        // necessary for the hdf5 lib
        hid_t memspace = H5Screate_simple(4, dims_local, NULL);
        int odd_offset;
        if (xlines > 1) {
            // this is theoretically the odd case but it has not been tested yet
            // so everything with odd should be ignored
            if((rank%num_io_parts) < int(dims[0]%xPerNode)) {
                offset[0] = ((xPerNode+1)*(rank%num_io_parts));
            } else {
                // offset of this x dim for this rank;
                // in most cases odd_offset will be 0
                odd_offset = (xPerNode+1)*(dims[0]%xPerNode);
                offset[0] = odd_offset +(xPerNode*((rank%num_io_parts)%xlines)); 
            }
        } else {
           offset[0] = 0;
        }
        
        if (ylines > 1) {
            if((rank%num_io_parts) < int(dims[1]%yPerNode)) {
                offset[1] = ((yPerNode+1)*(rank%num_io_parts));
             } else {
                // offset of the y dim for this rank
                odd_offset = (yPerNode+1)*(dims[1]%yPerNode);
                offset[1] = odd_offset + (yPerNode*((rank%num_io_parts)/ylines));
             }   
        } else {
            offset[1] = 0;
        }

        if (zlines > 1) {
            offset[2] = zPerNode*(rank%num_io_parts);
        } else {
            offset[2] = 0;
        }
        // I have channel dim splits 
        // all combos arent really tested
        //add the rest later
        offset[3] = 0; 
        count[0] = 1;
        count[1] = 1;
        count[2] = 1;
        count[3] = 1;
        // from an explanation of the hdf5 select_hyperslab:
        // start -> a starting location for the hyperslab
        // stride -> the number of elements to separate each element or block to be selected
        // count -> the number of elemenets or blocks to select along each dimension
        // block -> the size of the block selected from the dataspace 
        //hsize_t status;
        
        //todo add error checking
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, dims_local);
        H5Dread(h_data, H5T_NATIVE_SHORT, memspace, filespace, H5P_DEFAULT, data_out);
    }

    void hdf5_reader::load() {
       
        std::string dirpath = get_file_dir();    
        std::vector<std::string> file_list = get_filenames(dirpath);
        lbann_comm* l_comm = get_comm();
        const El::mpi::Comm & w_comm = l_comm->get_world_comm();
        MPI_Comm mpi_comm = w_comm.GetMPIComm();
        int world_rank = get_rank_in_world();
        int color = world_rank/dc::get_number_of_io_paritions(); 
        MPI_Comm_split(mpi_comm, color, world_rank, &m_comm);
        m_shuffled_indices.clear();
        m_shuffled_indices.resize(m_file_paths.size());
        std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
        std::cout<<"size after load " << m_shuffled_indices.size();
        
        select_subset_of_data();
    }
    bool hdf5_reader::fetch_label(Mat& Y, int data_id, int mb_idx) {
        return true;
    }
    bool hdf5_reader::fetch_datum(Mat& X, int data_id, int mb_idx) {
        prof_region_begin("fetch_datum", prof_colors[0], false);
        //TODO move this to load
	// put com into groups based off of the number of io splits
	// create a member variable
        double start = MPI_Wtime();
        lbann_comm* l_comm = get_comm();
        const El::mpi::Comm & w_comm = l_comm->get_world_comm();
        MPI_Comm mpi_comm = w_comm.GetMPIComm();
        int nprocs;
        MPI_Comm_size(mpi_comm, &nprocs); 
        if ((nprocs%dc::get_number_of_io_partitions()) !=0) {
            std::cerr<<"if other things have not been changed in the code this will not work for anything other than 4 procs a node \n";
        }
        int world_rank = get_rank_in_world(); 
        // for file in num files/ (num processes/number of split)
        // if we have 16 file and 4 processes and 4 splits then
        // each proc will see each file 
        //for(unsigned int nux =0; nux<(file_list.size()/(nprocs/dc::get_number_of_io_partitions())); nux++) {
            // math to figure out what file in the file list this proc should
            // currently be reading from
        double start_file = MPI_Wtime();
        //TODO: do I need this mod --> will world rank/paritions ever be greater than the number of files??
	    auto file = m_file_list[((world_rank/dc::get_number_of_io_partitions())+nux)%(m_file_list.size())];
            
        hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(fapl_id, mpi_comm, MPI_INFO_NULL); 
        hid_t h_file = H5Fopen(file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (h_file < 0) {
            throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + 
                                    " hdf5_reader::load() - can't open file : " + file);
        }

        // load in dataset
        hid_t h_data =  H5Dopen(h_file, HDF5_KEY_DATA.c_str(), H5P_DEFAULT);
        hid_t filespace = H5Dget_space(h_data);
        int rank1 = H5Sget_simple_extent_ndims(filespace);
        hsize_t dims[rank1];
        H5Sget_simple_extent_dims(filespace, dims, NULL);
            
        if (h_data < 0) {
            throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
                                        " hdf5_reader::load() - can't find hdf5 key : " + HDF5_KEY_DATA);
        }

        short int*& tmp = m_image_data[data_id];
        //TODO: add the int 16 stuff
	//TODO: change the indexing 
	// check if mb_idx needs to be changed to not be hard coded
	int adj_mb_idx = mb_idx+(rank%4);
        Mat X_v = El::View(X, El::IR(0,X.Height()), El::IR(adj_mb_idx, adj_mb_idx+1));

        DataType *dest = X_v.Buffer();
        //TODO Add a reference to X mat so only one read
	//will this work ? ?
        read_hdf5(h_data, filespace, world_rank, HDF5_KEY_DATA, dims, dest);
        //close data set
        H5Dclose(h_data);
        if (m_has_responses) {
            //TODO: move this to be a memeber variable act as a cache so we only have to open the file once
            h_data = H5Dopen(h_file, HDF5_KEY_RESPONSES.c_str(), H5P_DEFAULT);
            H5Dread(h_data, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_all_responses);
            H5Dclose(h_data);
        }
        H5Fclose(h_file);
        double end_file= MPI_Wtime();
        std::cerr<<"per file time " << end_file -start_file << "seconds to run. \n";
       }

        //TODO do i need this?
	// not if I pass a ref to X I dont think
	//this should be equal to num_nuerons/LBANN_NUM_IO_PARTITIONS
        //unsigned long int pixelcount = m_image_width*m_image_height*m_image_depth*m_image_num_channels;
  // #ifdef LBANN_DISTCONV_COSMOFLOW_KEEP_INT16
    //    std::memcpy(dest,data, sizeof(short)*pixelcount);
  // #else
    //    LBANN_OMP_PARALLEL_FOR
     //       for(int p = 0; p<pixelcount; p++) {
                //TODO what is m_scaling_factor_int16
     //           dest[p] = tmp[p] * m_scaling_factor_int16;
                // mash this with above
                    //X.Set(p, mb_idx,*tmp++);
     //       }
  // #endif
        //auto pixel_col = X(El::IR(0, X.Height()), El::IR(mb_idx, mb_idx+1));
        //std::vector<size_t> dims = {
          //  1ull,
          //  static_cast<size_t>(m_image_height),
          //  static_cast<size_t>(m_image_width)};
        //m_transform_pipeline.apply(pixel_col, dims);
        prof_region_end("fetch_datum", false);
        return true;
    }
    //get from a cached response
    bool hdf5_reader::fetch_response(Mat& Y, int data_id, int mb_idx) {
        prof_region_begin("fetch_response", prof_colors[0], false);
        Mat Y_v = El::View(Y, El::IR(0, Y.Height()), El::IR(mb_idx, mb_idx+1));
        //TODO: possibly 4 tho, python tells me its float64
        std::memcpy(Y_v.Buffer(), &m_all_responses,
            m_num_responses_features*8);
        prof_region_end("fetch_response", false);
        return true;
    } 
    
};
    
