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
//// data_reader_numpy_npz .hpp .cpp - generic_data_reader class for numpy .npz dataset
//////////////////////////////////////////////////////////////////////////////////
//
#include "lbann/data_readers/data_reader_hdf5.hpp"
#include <cstdio>
#include <string>
#include <fstream>
#include <unordered_set>
#include <iostream>
#include "dirent.h"
#include <cstring>

namespace lbann {
    const std::string hdf5_reader::HDF5_KEY_DATA = "full";
    const std::string hdf5_reader::HDF5_KEY_LABELS = "labels";
    const std::string hdf5_reader::HDF5_KEY_RESPONSES = "responses";

    hdf5_reader::hdf5_reader(const bool shuffle)
        : image_data_reader(shuffle) {
            set_defaults();
          }
    void hdf5_reader::set_linearized_image_size() {
        std::cout<< "in set_lin yay\n";
        m_image_linearized_size = m_image_width*m_image_depth*m_image_height*m_image_num_channels;
    }
    void hdf5_reader::set_defaults() {
        m_image_width = 512/2;
        m_image_height = 512/2;
        m_image_depth = 512;
        m_image_num_channels = 4;
        set_linearized_image_size();
    }
    // helper function, couldnt find this in the std lib
    bool file_ends_with(const std::string &mainStr, const std::string &toMatch)
    {
        return (mainStr.size() >= toMatch.size() &&
                mainStr.compare(mainStr.size() - toMatch.size(), toMatch.size(), toMatch) == 0);
    }
    // collect all files names in a directory, ignore all files that don't end in .hdf5
    // this should possibly be changes to "or .h5" as I think that is a valid ending to hdf5 files
    std::vector<std::string> get_filenames(std::string dir_path) {
        std::vector<std::string> file_names;
        DIR *dir = opendir(dir_path.c_str());
        struct dirent *entry;
        std::string file_ending = ".hdf5";
        while ((entry = readdir(dir)) != NULL) {
            std::string temp_path = dir_path;
            std::string entry_name = entry->d_name;

            if(file_ends_with(entry_name, file_ending)) {
                file_names.push_back(temp_path.append(entry_name));                                                 
            }
        }
        closedir(dir);
        return file_names;
    }

    void hdf5_reader::read_hdf5(hsize_t h_data, hsize_t filespace, int rank, std::string key, hsize_t* dims) {
        // this is the splits, right now it is hard coded to split along the y and x
        int ylines = 2;
        int xlines = 2;
        int zlines = 1;
        int channellines = 1;

        // todo: when taking care of the odd case this cant be an int
        int xPerNode = dims[0]/xlines;
        int yPerNode = dims[1]/ylines;
        int zPerNode = dims[2]/zlines;
        int cPerNode = dims[3]/channellines;
       
        short int data_out[xPerNode*yPerNode*zPerNode*cPerNode]; 
        hsize_t dims_local[4];
        hsize_t offset[4];
        hsize_t count[4];
        dims_local[0] = xPerNode;
        dims_local[1] = yPerNode;
        dims_local[2] = zPerNode;
        dims_local[3] = cPerNode;
        hid_t memspace = H5Screate_simple(4, dims_local, NULL);

        if (xlines > 1) {
            // this is theoretically the odd case but it has not been tested yet
            if((rank%4) < int(dims[0]%xPerNode)) {
                offset[0] = ((xPerNode+1)*(rank%2));
                xPerNode++;
            } else {
                offset[0] = ((xPerNode+1)*(dims[0]%xPerNode))+(xPerNode*(((rank%4)-(dims[0]%xPerNode))%xlines)); 
            }
        } else {
           offset[0] = 0;
        }
        
        if (ylines > 1) {
            if((rank%4) < int(dims[1]%yPerNode)) {
                offset[1] = ((yPerNode+1)*(rank%2));
                yPerNode++;
             } else {    
                offset[1] = ((yPerNode+1)*(dims[1]%yPerNode)) + (yPerNode*(((rank%4)-(dims[1]%yPerNode))/ylines));                                                   }   
        } else {
            offset[1] = 0;
        }

        //add the rest later
        offset[2] = 0;
        offset[3] = 0; 
        count[0] = 1;
        count[1] = 1;
        count[2] = 1;
        count[3] = 1;
        //std::cout<< " offset " <<offset[0] <<" " << offset[1] << " " << offset[2] << " " <<offset[3] << "\n";
        // start -> a starting location for the hyperslab
        // stride -> the number of elements to separate each element or block to be selected
        // count -> the number of elemenets or blocks to select along each dimension
        // block -> the size of the block selected from the dataspace 
        //hsize_t status;
        
        //todo add error checking
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, dims_local);
        H5Dread(h_data, H5T_NATIVE_SHORT, memspace, filespace, H5P_DEFAULT, data_out);
        
        m_image_data.push_back(&(*data_out));
    }

    void hdf5_reader::load() {
       
        std::string dirpath = get_file_dir();    
        std::vector<std::string> file_list = get_filenames(dirpath);
        
        double start = MPI_Wtime();
        lbann_comm* l_comm = get_comm();
        const El::mpi::Comm & w_comm = l_comm->get_world_comm();
        MPI_Comm mpi_comm = w_comm.GetMPIComm();
        int nprocs;
        MPI_Comm_size(mpi_comm, &nprocs);
        std::cout<< "nprocs " << nprocs<< "\n"; 
        if ((nprocs%4) !=0) {
            std::cerr<<"if other things have not been changed in the code this will not work for anything other than 4 procs a node \n";
        }
        int world_rank = get_rank_in_world();
        std::cout<<" world rank " << world_rank <<"\n"; 
        
        for(unsigned int nux =0; nux<(file_list.size()/(nprocs/4)); nux++) {
            auto file = file_list[((world_rank/4)+nux)%(file_list.size())];
            hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
            H5Pset_fapl_mpio(fapl_id, mpi_comm, MPI_INFO_NULL); 
            hid_t h_file = H5Fopen(file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            if (h_file < 0) {
                throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + 
                                    " hdf5_reader::load() - can't open file : " + file);
            }
     
            //std::vector<std::tuple<const bool, const std::string, hid_t &>> hdf5LoadList;
            // in other readers:
            // m_image_data[0] is label
            // m_image_data[1] is image
            // might want to change the structure once I know what the labels are

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
            
            read_hdf5(h_data, filespace, world_rank, HDF5_KEY_DATA, dims);
            
            H5Dclose(h_data);
            H5Fclose(h_file);
       }
        m_shuffled_indices.clear();
        std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
        m_shuffled_indices.resize(m_image_data.size());
        double end = MPI_Wtime();
        std::cerr<< "Num Files " << file_list.size() << "\n";
        std::cerr<< "Rank " << world_rank << " \n";
        std::cerr<< "The process took " << end - start << " seconds to run. \n";


    }
    bool hdf5_reader::fetch_label(Mat& Y, int data_id, int mb_idx) {
        return true;
    }
    bool hdf5_reader::fetch_datum(Mat& X, int data_id, int mb_idx) {
        int pixelcount = m_image_width*m_image_height*m_image_depth;
        short int*& tmp = m_image_data[data_id];
        for(int p = 0; p<pixelcount; p++) {
            X.Set(p, mb_idx,*tmp++);
        }
        auto pixel_col = X(El::IR(0, X.Height()), El::IR(mb_idx, mb_idx+1));
        std::vector<size_t> dims = {
            1ull,
            static_cast<size_t>(m_image_height),
            static_cast<size_t>(m_image_width)};
        m_transform_pipeline.apply(pixel_col, dims);
        return true;
    }
    bool hdf5_reader::fetch_response(Mat& Y, int data_id, int mb_idx) {
        return true;
    } 
    
};
    
